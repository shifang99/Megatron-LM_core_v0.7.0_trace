# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import contextlib
from typing import Callable, Iterator, List, Optional, Union

import torch
from torch.autograd.variable import Variable

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import p2p_communication
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
from megatron.core.utils import get_attr_wrapped_model, get_model_config, get_model_type

# Types
Shape = Union[List[int], torch.Size]


def get_forward_backward_func():
    """Retrieves the appropriate forward_backward function given the
    configuration of parallel_state.

    Returns a function that will perform all of the forward and
    backward passes of the model given the pipeline model parallel
    world size and virtual pipeline model parallel world size in the
    global parallel_state.

    Note that if using sequence parallelism, the sequence length component of
    the tensor shape is updated to original_sequence_length /
    tensor_model_parallel_world_size.

    The function returned takes the following arguments:

    forward_step_func (required): A function that takes a data
        iterator and a model as its arguments and return the model's
        forward output and the loss function. The loss function should
        take one torch.Tensor and return a torch.Tensor of loss and a
        dictionary of string -> torch.Tensor.

        A third argument, checkpoint_activations_microbatch, indicates
        that the activations for this microbatch should be
        checkpointed. A None value for this argument indicates that
        the default from the configuration should be used. This is
        used when the
        num_microbatches_with_partial_activation_checkpoints is used.

        For example:

        def loss_func(loss_mask, output_tensor):
            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

            # Reduce loss for logging.
            averaged_loss = average_losses_across_data_parallel_group([loss])

            return loss, {'lm loss': averaged_loss[0]}

        def forward_step(data_iterator, model):
            data, loss_mask = next(data_iterator)
            output = model(data)
            return output, partial(loss_func, loss_mask)


        forward_backward_func(forward_step_func=forward_step, ...)


    data_iterator (required): an iterator over the data, will be
        passed as is to forward_step_func. Expected to be a list of
        iterators in the case of interleaved pipeline parallelism.

    model (required): the actual model. Expected to be a list of modules in the case of interleaved
        pipeline parallelism. Must be a (potentially wrapped) megatron.core.models.MegatronModule.

    num_microbatches (int, required):
        The number of microbatches to go through

    seq_length (int, required): Sequence length of the current global batch. If this is a dual-stack
        transformer, this is the encoder's sequence length. This is ignored if variable_seq_lengths
        in the config is True. Otherwise, each microbatch in the current global batch size must use
        this sequence length.

    micro_batch_size (int, required): The number of sequences in a microbatch.

    decoder_seq_length (int, optional): The sequence length for the decoder in a dual-stack
        transformer. This is ignored for a single-stack transformer.

    forward_only (optional, default = False): Perform only the forward step

    collect_non_loss_data (optional, bool, default=False): TODO

    first_val_step (bool, optional): Is the first step of the validation phase. Used by
        Transformer Engine modules to only update their fp8 weights only on the first validation step.

    """
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()# trace_info : t_17657, t_20843, t_24029
    if pipeline_model_parallel_size > 1:                                       # trace_info : t_17662, t_20848, t_24034
        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            forward_backward_func = forward_backward_pipelining_with_interleaving
        else:
            forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = forward_backward_no_pipelining                 # trace_info : t_17663, t_20849, t_24035
    return forward_backward_func                                               # trace_info : t_17664, t_20850, t_24036


def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
    '''Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.

    This method should be called right after the output tensor has been
    sent to the next pipeline stage. At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.
    '''
    if (out is None) or (not deallocate_pipeline_outputs):
        return
    assert isinstance(out, torch.Tensor), "expected Tensor, found %s." % type(out).__name__
    assert out._base is None, "counter-productive to free a view of another tensor."
    out.data = torch.empty((1,), device=out.device, dtype=out.dtype,)


def custom_backward(output, grad_output):
    '''Directly call C++ autograd engine.

    To make the 'deallocate_output_tensor' (above) optimization work, the C++
    autograd engine must be called directly, bypassing Pytorch's
    torch.autograd.backward. Pytorch's 'backward' checks that the output and
    grad have the same shape, while C++'s 'backward' does not.
    '''

    assert output.numel() == 1, "output should be pseudo-'freed' in schedule, to optimize memory"# trace_info : t_18948, t_22134, t_25320
    assert isinstance(output, torch.Tensor), "output == '%s'." % type(output).__name__# trace_info : t_18949, t_22135, t_25321
    assert isinstance(grad_output, (torch.Tensor, type(None))), (              # trace_info : t_18950, t_22136, t_25322
        "grad_output == '%s'." % type(grad_output).__name__
    )

    # Handle scalar output
    if grad_output is None:                                                    # trace_info : t_18951, t_22137, t_25323
        assert output.numel() == 1, "implicit grad requires scalar output."    # trace_info : t_18952, t_22138, t_25324
        grad_output = torch.ones_like(output, memory_format=torch.preserve_format,)# trace_info : t_18953, t_22139, t_25325

    # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
    Variable._execution_engine.run_backward(                                   # trace_info : t_18954, t_18962, t_22140, t_22148, t_25326, ...
        tensors=(output,),                                                     # trace_info : t_18955, t_22141, t_25327
        grad_tensors=(grad_output,),                                           # trace_info : t_18956, t_22142, t_25328
        keep_graph=False,                                                      # trace_info : t_18957, t_22143, t_25329
        create_graph=False,                                                    # trace_info : t_18958, t_22144, t_25330
        inputs=tuple(),                                                        # trace_info : t_18959, t_22145, t_25331
        allow_unreachable=True,                                                # trace_info : t_18960, t_22146, t_25332
        accumulate_grad=True,                                                  # trace_info : t_18961, t_22147, t_25333
    )


def set_current_microbatch(model, microbatch_id):
    decoder_exists = True                                                      # trace_info : t_17863, t_21049, t_24235
    decoder = None                                                             # trace_info : t_17864, t_21050, t_24236
    try:                                                                       # trace_info : t_17865, t_21051, t_24237
        decoder = get_attr_wrapped_model(model, "decoder")                     # trace_info : t_17866, t_21052, t_24238
    except RuntimeError:
        decoder_exists = False
    if decoder_exists and decoder is not None:                                 # trace_info : t_17882, t_21068, t_24254
        decoder.current_microbatch = microbatch_id                             # trace_info : t_17883, t_21069, t_24255


def forward_step(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    forward_data_store,
    config,
    collect_non_loss_data=False,
    checkpoint_activations_microbatch=None,
    is_first_microbatch=False,
    current_microbatch=None,
):

    """Forward step for passed-in model.

    If first stage, input tensor is obtained from data_iterator, otherwise
    passed-in input_tensor is used.

    Returns output tensor."""
    if config.timers is not None:                                              # trace_info : t_17740, t_20926, t_24112
        config.timers('forward-compute', log_level=2).start()                  # trace_info : t_17741, t_20927, t_24113

    if is_first_microbatch and hasattr(model, 'set_is_first_microbatch'):      # trace_info : t_17748, t_20934, t_24120
        model.set_is_first_microbatch()                                        # trace_info : t_17749, t_20935, t_24121
    if current_microbatch is not None:                                         # trace_info : t_17861, t_21047, t_24233
        set_current_microbatch(model, current_microbatch)                      # trace_info : t_17862, t_21048, t_24234

    unwrap_output_tensor = False                                               # trace_info : t_17884, t_21070, t_24256
    if not isinstance(input_tensor, list):                                     # trace_info : t_17885, t_21071, t_24257
        input_tensor = [input_tensor]                                          # trace_info : t_17886, t_21072, t_24258
        unwrap_output_tensor = True                                            # trace_info : t_17887, t_21073, t_24259

    set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")       # trace_info : t_17888, t_21074, t_24260
    set_input_tensor(input_tensor)                                             # trace_info : t_17900, t_21086, t_24272

    if config.enable_autocast:                                                 # trace_info : t_17906, t_21092, t_24278
        context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()                             # trace_info : t_17907, t_21093, t_24279
    with context_manager:                                                      # trace_info : t_17908, t_18831, t_21094, t_22017, t_24280, ...
        if checkpoint_activations_microbatch is None:                          # trace_info : t_17909, t_21095, t_24281
            output_tensor, loss_func = forward_step_func(data_iterator, model) # trace_info : t_17910, t_21096, t_24282
        else:
            output_tensor, loss_func = forward_step_func(
                data_iterator, model, checkpoint_activations_microbatch
            )

    num_tokens = torch.tensor(0, dtype=torch.int)                              # trace_info : t_18832, t_22018, t_25204
    if parallel_state.is_pipeline_last_stage():                                # trace_info : t_18833, t_22019, t_25205
        if not collect_non_loss_data:                                          # trace_info : t_18848, t_22034, t_25220
            outputs = loss_func(output_tensor)                                 # trace_info : t_18849, t_22035, t_25221
            if len(outputs) == 3:                                              # trace_info : t_18874, t_22060, t_25246
                output_tensor, num_tokens, loss_reduced = outputs              # trace_info : t_18875, t_22061, t_25247
                if not config.calculate_per_token_loss:                        # trace_info : t_18876, t_22062, t_25248
                    output_tensor /= num_tokens                                # trace_info : t_18877, t_22063, t_25249
                    output_tensor /= num_microbatches                          # trace_info : t_18878, t_22064, t_25250
            else:
                # preserve legacy loss averaging behavior (ie, over the number of microbatches)
                assert len(outputs) == 2
                output_tensor, loss_reduced = outputs
                output_tensor /= num_microbatches
            forward_data_store.append(loss_reduced)                            # trace_info : t_18879, t_22065, t_25251
        else:
            data = loss_func(output_tensor, non_loss_data=True)
            forward_data_store.append(data)

    if config.timers is not None:                                              # trace_info : t_18880, t_22066, t_25252
        config.timers('forward-compute').stop()                                # trace_info : t_18881, t_22067, t_25253

    # Set the loss scale for the auxiliary loss of the MoE layer.
    # Since we use a trick to do backward on the auxiliary loss, we need to set the scale explicitly.
    if hasattr(config, 'num_moe_experts') and config.num_moe_experts is not None:# trace_info : t_18889, t_22075, t_25261
        # Calculate the loss scale based on the grad_scale_func if available, else default to 1.
        loss_scale = (
            config.grad_scale_func(torch.tensor(1.0, device=output_tensor.device))
            if config.grad_scale_func is not None
            else torch.tensor(1.0)
        )
        # Set the loss scale
        MoEAuxLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)

    # If T5 model (or other model with encoder and decoder)
    # and in decoder stack, then send encoder_hidden_state
    # downstream as well.
    model_type = get_model_type(model)                                         # trace_info : t_18890, t_22076, t_25262
    if (                                                                       # trace_info : t_18914, t_22100, t_25286
        parallel_state.is_pipeline_stage_after_split()                         # trace_info : t_18907, t_22093, t_25279
        and model_type == ModelType.encoder_and_decoder                        # trace_info : t_18915, t_22101, t_25287
    ):
        return [output_tensor, input_tensor[-1]], num_tokens

    if unwrap_output_tensor:                                                   # trace_info : t_18916, t_22102, t_25288
        return output_tensor, num_tokens                                       # trace_info : t_18917, t_22103, t_25289
    return [output_tensor], num_tokens


def backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.

    if config.timers is not None:                                              # trace_info : t_18921, t_22107, t_25293
        config.timers('backward-compute', log_level=2).start()                 # trace_info : t_18922, t_22108, t_25294

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False                                           # trace_info : t_18929, t_22115, t_25301
    if not isinstance(input_tensor, list):                                     # trace_info : t_18930, t_22116, t_25302
        input_tensor = [input_tensor]                                          # trace_info : t_18931, t_22117, t_25303
        unwrap_input_tensor_grad = True                                        # trace_info : t_18932, t_22118, t_25304
    for x in input_tensor:                                                     # trace_info : t_18933, t_18935, t_22119, t_22121, t_25305, ...
        if x is not None:                                                      # trace_info : t_18934, t_22120, t_25306
            x.retain_grad()

    if not isinstance(output_tensor, list):                                    # trace_info : t_18936, t_22122, t_25308
        output_tensor = [output_tensor]                                        # trace_info : t_18937, t_22123, t_25309
    if not isinstance(output_tensor_grad, list):                               # trace_info : t_18938, t_22124, t_25310
        output_tensor_grad = [output_tensor_grad]                              # trace_info : t_18939, t_22125, t_25311

    # Backward pass.
    if output_tensor_grad[0] is None and config.grad_scale_func is not None:   # trace_info : t_18940, t_22126, t_25312
        output_tensor[0] = config.grad_scale_func(output_tensor[0])            # trace_info : t_18941, t_22127, t_25313

    if config.deallocate_pipeline_outputs:                                     # trace_info : t_18946, t_22132, t_25318
        custom_backward(output_tensor[0], output_tensor_grad[0])               # trace_info : t_18947, t_22133, t_25319
    else:
        torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])

    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]                                                 # trace_info : t_18963, t_22149, t_25335
    if input_tensor is not None:                                               # trace_info : t_18964, t_22150, t_25336
        input_tensor_grad = []                                                 # trace_info : t_18965, t_22151, t_25337
        for x in input_tensor:                                                 # trace_info : t_18966, t_18969, t_22152, t_22155, t_25338, ...
            if x is None:                                                      # trace_info : t_18967, t_22153, t_25339
                input_tensor_grad.append(None)                                 # trace_info : t_18968, t_22154, t_25340
            else:
                input_tensor_grad.append(x.grad)

    # Handle single skip connection if it exists (encoder_hidden_state in
    # model with encoder and decoder).
    if (
        parallel_state.get_pipeline_model_parallel_world_size() > 1            # trace_info : t_18970, t_22156, t_25342
        and parallel_state.is_pipeline_stage_after_split()
        and model_type == ModelType.encoder_and_decoder
    ):
        if output_tensor_grad[1] is not None:
            input_tensor_grad[-1].add_(output_tensor_grad[1])
    if unwrap_input_tensor_grad:                                               # trace_info : t_18975, t_22161, t_25347
        input_tensor_grad = input_tensor_grad[0]                               # trace_info : t_18976, t_22162, t_25348

    if config.timers is not None:                                              # trace_info : t_18977, t_22163, t_25349
        config.timers('backward-compute').stop()                               # trace_info : t_18978, t_22164, t_25350

    return input_tensor_grad                                                   # trace_info : t_18986, t_22172, t_25358


def check_first_val_step(first_val_step, forward_only, cond):
    if (first_val_step is not None) and forward_only:                          # trace_info : t_17736, t_20922, t_24108
        return first_val_step and cond
    else:
        return cond                                                            # trace_info : t_17737, t_20923, t_24109


def forward_backward_no_pipelining(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,  # unused
    micro_batch_size: int,  # unused
    decoder_seq_length: int = None,  # unused
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """Run forward and backward passes with no pipeline parallelism
    (no inter-stage communication).

    Returns dictionary with losses.


    See get_forward_backward_func() for argument details
    """

    if isinstance(model, list):                                                # trace_info : t_17677, t_20863, t_24049
        assert len(model) == 1, "non-pipeline-parallel schedule does not support model chunking"# trace_info : t_17678, t_20864, t_24050
        model = model[0]                                                       # trace_info : t_17679, t_20865, t_24051
    if isinstance(data_iterator, list):                                        # trace_info : t_17680, t_20866, t_24052
        assert (
            len(data_iterator) == 1
        ), "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)                                           # trace_info : t_17681, t_20867, t_24053
    if config.timers is not None:                                              # trace_info : t_17690, t_20876, t_24062
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)# trace_info : t_17691, t_20877, t_24063

    no_sync_func = config.no_sync_func                                         # trace_info : t_17698, t_20884, t_24070
    if no_sync_func is None:                                                   # trace_info : t_17699, t_20885, t_24071
        no_sync_func = contextlib.nullcontext                                  # trace_info : t_17700, t_20886, t_24072

    model_type = get_model_type(model)                                         # trace_info : t_17701, t_20887, t_24073

    forward_data_store = []                                                    # trace_info : t_17718, t_20904, t_24090
    input_tensor, output_tensor_grad = None, None                              # trace_info : t_17719, t_20905, t_24091
    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()                 # trace_info : t_17720, t_20906, t_24092
    with no_sync_func():                                                       # trace_info : t_17721, t_17723, t_20907, t_20909, t_24093, ...
        for i in range(num_microbatches - 1):                                  # trace_info : t_17722, t_20908, t_24094
            output_tensor, num_tokens = forward_step(
                forward_step_func,
                data_iterator,
                model,
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                collect_non_loss_data,
                is_first_microbatch=check_first_val_step(first_val_step, forward_only, i == 0),
                current_microbatch=i,
            )
            total_num_tokens += num_tokens.item()
            if not forward_only:
                backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)

    # Run computation for last microbatch out of context handler (want to
    # synchronize gradients).
    output_tensor, num_tokens = forward_step(                                  # trace_info : t_17724, t_17739, t_20910, t_20925, t_24096, ...
        forward_step_func,                                                     # trace_info : t_17725, t_20911, t_24097
        data_iterator,                                                         # trace_info : t_17726, t_20912, t_24098
        model,                                                                 # trace_info : t_17727, t_20913, t_24099
        num_microbatches,                                                      # trace_info : t_17728, t_20914, t_24100
        input_tensor,                                                          # trace_info : t_17729, t_20915, t_24101
        forward_data_store,                                                    # trace_info : t_17730, t_20916, t_24102
        config,                                                                # trace_info : t_17731, t_20917, t_24103
        collect_non_loss_data,                                                 # trace_info : t_17732, t_20918, t_24104
        is_first_microbatch=check_first_val_step(                              # trace_info : t_17733, t_17735, t_20919, t_20921, t_24105, ...
            first_val_step, forward_only, num_microbatches == 1                # trace_info : t_17734, t_20920, t_24106
        ),
        current_microbatch=num_microbatches - 1,                               # trace_info : t_17738, t_20924, t_24110
    )
    total_num_tokens += num_tokens.item()                                      # trace_info : t_18918, t_22104, t_25290

    if not forward_only:                                                       # trace_info : t_18919, t_22105, t_25291
        backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)# trace_info : t_18920, t_22106, t_25292

    if config.finalize_model_grads_func is not None and not forward_only:      # trace_info : t_18987, t_22173, t_25359
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism and layernorm all-reduce for sequence parallelism).
        config.finalize_model_grads_func(                                      # trace_info : t_18988, t_18990, t_22174, t_22176, t_25360, ...
            [model], total_num_tokens if config.calculate_per_token_loss else None# trace_info : t_18989, t_22175, t_25361
        )

    if config.timers is not None:                                              # trace_info : t_19112, t_22298, t_25484
        config.timers('forward-backward').stop()                               # trace_info : t_19113, t_22299, t_25485

    return forward_data_store                                                  # trace_info : t_19121, t_22307, t_25493


def forward_backward_pipelining_with_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """Run interleaved 1F1B schedule (model split into model chunks), with
    communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    assert isinstance(model, list), "interleaved pipeline parallelism expected model chunking"
    assert all(isinstance(chunk, torch.nn.Module) for chunk in model), "invalid model chunking"
    assert isinstance(
        data_iterator, list
    ), "interleaved pipeline parallelism expected each model chunk to have a data iterator"

    config = get_model_config(model[0])
    if config.overlap_p2p_comm and config.batch_p2p_comm:
        raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if isinstance(no_sync_func, list):

        def multi_no_sync():
            stack = contextlib.ExitStack()
            for model_chunk_no_sync_func in config.no_sync_func:
                stack.enter_context(model_chunk_no_sync_func())
            return stack

        no_sync_func = multi_no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    if config.grad_sync_func is not None and not isinstance(config.grad_sync_func, list):
        config.grad_sync_func = [config.grad_sync_func for _ in model]

    if config.param_sync_func is not None and not isinstance(config.param_sync_func, list):
        config.param_sync_func = [config.param_sync_func for _ in model]

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Model chunk IDs with synchronized grads
    synchronized_model_chunks = set()

    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()

    forward_data_store = []
    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]

    pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()

    if num_microbatches % pipeline_parallel_size != 0:
        msg = f'number of microbatches ({num_microbatches}) is not divisible by '
        msg += f'pipeline-model-parallel-size ({pipeline_parallel_size}) '
        msg += 'when using interleaved schedule'
        raise RuntimeError(msg)

    model_type = get_model_type(model[0])
    if model_type == ModelType.encoder_and_decoder:
        raise RuntimeError("Interleaving is not supported with an encoder and decoder model.")

    if decoder_seq_length is not None and decoder_seq_length != seq_length:
        raise RuntimeError(
            "Interleaving is not supported with a different decoder sequence length."
        )

    tensor_shape = [seq_length, micro_batch_size, config.hidden_size]
    tensor_shape[0] = tensor_shape[0] // parallel_state.get_context_parallel_world_size()
    if config.sequence_parallel:
        tensor_shape[0] = tensor_shape[0] // parallel_state.get_tensor_model_parallel_world_size()

    # Compute number of warmup and remaining microbatches.
    num_model_chunks = len(model)
    total_num_microbatches = num_microbatches * num_model_chunks
    all_warmup_microbatches = False
    if forward_only:
        num_warmup_microbatches = total_num_microbatches
    else:
        # Run all forward passes and then all backward passes if number of
        # microbatches is just the number of pipeline stages.
        # Otherwise, perform (num_model_chunks-1)*pipeline_parallel_size on
        # all workers, followed by more microbatches after depending on
        # stage ID (more forward passes for earlier stages, later stages can
        # immediately start with 1F1B).
        if num_microbatches == pipeline_parallel_size:
            num_warmup_microbatches = total_num_microbatches
            all_warmup_microbatches = True
        else:
            num_warmup_microbatches = (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
            num_warmup_microbatches += (num_model_chunks - 1) * pipeline_parallel_size
            num_warmup_microbatches = min(num_warmup_microbatches, total_num_microbatches)
    num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    # Synchronize params for first two model chunks
    if config.param_sync_func is not None:
        config.param_sync_func[0](model[0].parameters())
        config.param_sync_func[1](model[1].parameters())

    def get_model_chunk_id(microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        microbatch_id_in_group = microbatch_id % (pipeline_parallel_size * num_model_chunks)
        model_chunk_id = microbatch_id_in_group // pipeline_parallel_size
        if not forward:
            model_chunk_id = num_model_chunks - model_chunk_id - 1
        return model_chunk_id

    def get_microbatch_id_in_model_chunk(iteration_id, forward):
        """Helper method to get the microbatch_id within model chunk given the iteration number."""
        assert forward
        iteration_group_id = iteration_id // (pipeline_parallel_size * num_model_chunks)
        microbatch_id_in_model_chunk = (iteration_group_id * pipeline_parallel_size) + (
            iteration_id % pipeline_parallel_size
        )
        return microbatch_id_in_model_chunk

    def is_first_microbatch_for_model_chunk(microbatch_id: int) -> bool:
        """Check if an iteration is the first for a model chunk."""
        microbatch_group_size = pipeline_parallel_size * num_model_chunks
        num_microbatch_groups = total_num_microbatches // microbatch_group_size
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        if microbatch_group_id == 0:
            return microbatch_id_in_group % pipeline_parallel_size == 0
        else:
            return False

    def is_last_microbatch_for_model_chunk(microbatch_id: int) -> bool:
        """Check if an iteration is the last for a model chunk."""
        microbatch_group_size = pipeline_parallel_size * num_model_chunks
        num_microbatch_groups = total_num_microbatches // microbatch_group_size
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        if microbatch_group_id == num_microbatch_groups - 1:
            return microbatch_id_in_group % pipeline_parallel_size == pipeline_parallel_size - 1
        else:
            return False

    def forward_step_helper(microbatch_id, current_microbatch, checkpoint_activations_microbatch):
        """Helper method to run forward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        forward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=True)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # launch param synchronization for next model chunk
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.param_sync_func is not None:
            param_sync_microbatch_id = microbatch_id + pipeline_parallel_rank
            if (
                param_sync_microbatch_id < total_num_microbatches
                and is_first_microbatch_for_model_chunk(param_sync_microbatch_id)
            ):
                param_sync_chunk_id = get_model_chunk_id(param_sync_microbatch_id, forward=True) + 1
                if 1 < param_sync_chunk_id < num_model_chunks:
                    config.param_sync_func[param_sync_chunk_id](
                        model[param_sync_chunk_id].parameters()
                    )

        # forward step
        if parallel_state.is_pipeline_first_stage():
            if len(input_tensors[model_chunk_id]) == len(output_tensors[model_chunk_id]):
                input_tensors[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id][-1]

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator[model_chunk_id],
            model[model_chunk_id],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(
                first_val_step, forward_only, is_first_microbatch_for_model_chunk(microbatch_id),
            ),
            current_microbatch=current_microbatch,
        )
        output_tensors[model_chunk_id].append(output_tensor)

        nonlocal total_num_tokens
        total_num_tokens += num_tokens.item()

        # if forward-only, no need to save tensors for a backward pass
        if forward_only:
            input_tensors[model_chunk_id].pop()
            output_tensors[model_chunk_id].pop()

        return output_tensor

    def backward_step_helper(microbatch_id):
        """Helper method to run backward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        backward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=False)
        parallel_state.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # launch grad synchronization (default)
        if config.grad_sync_func is None and is_last_microbatch_for_model_chunk(microbatch_id):
            enable_grad_sync()
            synchronized_model_chunks.add(model_chunk_id)

        if parallel_state.is_pipeline_last_stage():
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)
        input_tensor_grad = backward_step(
            input_tensor, output_tensor, output_tensor_grad, model_type, config
        )

        # launch grad synchronization (custom grad sync)
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.grad_sync_func is not None:
            grad_sync_microbatch_id = microbatch_id - pipeline_parallel_rank
            if grad_sync_microbatch_id >= 0 and is_last_microbatch_for_model_chunk(
                grad_sync_microbatch_id
            ):
                grad_sync_chunk_id = get_model_chunk_id(grad_sync_microbatch_id, forward=False)
                enable_grad_sync()
                config.grad_sync_func[grad_sync_chunk_id](model[grad_sync_chunk_id].parameters())
                synchronized_model_chunks.add(grad_sync_chunk_id)
        disable_grad_sync()

        return input_tensor_grad

    # Run warmup forward passes.
    parallel_state.set_virtual_pipeline_model_parallel_rank(0)
    input_tensors[0].append(p2p_communication.recv_forward(tensor_shape, config))

    fwd_wait_handles = None
    bwd_wait_handles = None

    for k in range(num_warmup_microbatches):

        if fwd_wait_handles is not None:
            for req in fwd_wait_handles:
                req.wait()

        cur_model_chunk_id = get_model_chunk_id(k, forward=True)
        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                k % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        current_microbatch = get_microbatch_id_in_model_chunk(k, forward=True)
        output_tensor = forward_step_helper(
            k, current_microbatch, checkpoint_activations_microbatch
        )

        # Determine if tensor should be received from previous stage.
        next_forward_model_chunk_id = get_model_chunk_id(k + 1, forward=True)
        recv_prev = True
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            if next_forward_model_chunk_id == 0:
                recv_prev = False
        if k == (total_num_microbatches - 1):
            recv_prev = False

        # Don't send tensor downstream if on last stage.
        if parallel_state.is_pipeline_last_stage():
            output_tensor = None

        # Send and receive tensors as appropriate (send tensors computed
        # in this iteration; receive tensors for next iteration).
        if not config.overlap_p2p_comm:
            if (
                k == (num_warmup_microbatches - 1)
                and not forward_only
                and not all_warmup_microbatches
            ):
                input_tensor_grad = None
                recv_next = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False
                (
                    input_tensor,
                    output_tensor_grad,
                ) = p2p_communication.send_forward_backward_recv_forward_backward(
                    output_tensor,
                    input_tensor_grad,
                    recv_prev=recv_prev,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    config=config,
                )
                output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
            else:
                input_tensor = p2p_communication.send_forward_recv_forward(
                    output_tensor, recv_prev=recv_prev, tensor_shape=tensor_shape, config=config
                )
            input_tensors[next_forward_model_chunk_id].append(input_tensor)
        else:
            input_tensor, fwd_wait_handles = p2p_communication.send_forward_recv_forward(
                output_tensor,
                recv_prev=recv_prev,
                tensor_shape=tensor_shape,
                config=config,
                overlap_p2p_comm=True,
            )

            if (
                k == (num_warmup_microbatches - 1)
                and not forward_only
                and not all_warmup_microbatches
            ):
                input_tensor_grad = None
                recv_next = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    recv_next = False

                (
                    output_tensor_grad,
                    bwd_wait_handles,
                ) = p2p_communication.send_backward_recv_backward(
                    input_tensor_grad,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    config=config,
                    overlap_p2p_comm=True,
                )

                output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
            input_tensors[next_forward_model_chunk_id].append(input_tensor)

        deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    # Run 1F1B in steady state.
    for k in range(num_microbatches_remaining):
        # Forward pass.
        forward_k = k + num_warmup_microbatches

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                forward_k % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        cur_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
        current_microbatch = get_microbatch_id_in_model_chunk(forward_k, forward=True)
        if config.overlap_p2p_comm:
            if fwd_wait_handles is not None:
                for req in fwd_wait_handles:
                    req.wait()

            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            output_tensor = forward_step_helper(
                forward_k, current_microbatch, checkpoint_activations_microbatch
            )

            # Determine if current stage has anything to send in either direction,
            # otherwise set tensor to None.
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)

            # Last virtual stage no activation tensor to send
            if parallel_state.is_pipeline_last_stage():
                output_tensor = None

            # Determine if peers are sending, and where in data structure to put
            # received tensors.
            recv_prev = True
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                # First stage is ahead of last stage by (pipeline_parallel_size - 1).
                next_forward_model_chunk_id = get_model_chunk_id(
                    forward_k - (pipeline_parallel_size - 1), forward=True
                )
                if next_forward_model_chunk_id == (num_model_chunks - 1):
                    recv_prev = False
                next_forward_model_chunk_id += 1
            else:
                next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1, forward=True)

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.
            if k == (num_microbatches_remaining - 1):
                recv_prev = False

            # Send activation tensor to the next stage and receive activation tensor from the
            # previous stage
            input_tensor, fwd_wait_handles = p2p_communication.send_forward_recv_forward(
                output_tensor,
                recv_prev=recv_prev,
                tensor_shape=tensor_shape,
                config=config,
                overlap_p2p_comm=True,
            )
            # assert fwd_wait_handles is not None

            if bwd_wait_handles is not None:
                for req in bwd_wait_handles:
                    req.wait()

            # Backward pass.
            backward_k = k
            input_tensor_grad = backward_step_helper(backward_k)

            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)

            # First virtual stage no activation gradient tensor to send
            if parallel_state.is_pipeline_first_stage():
                input_tensor_grad = None

            # Determine if the current virtual stage has an activation gradient tensor to receive
            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
                next_backward_model_chunk_id = get_model_chunk_id(
                    backward_k - (pipeline_parallel_size - 1), forward=False
                )
                if next_backward_model_chunk_id == 0:
                    recv_next = False
                next_backward_model_chunk_id -= 1
            else:
                next_backward_model_chunk_id = get_model_chunk_id(backward_k + 1, forward=False)

            output_tensor_grad, bwd_wait_handles = p2p_communication.send_backward_recv_backward(
                input_tensor_grad,
                recv_next=recv_next,
                tensor_shape=tensor_shape,
                config=config,
                overlap_p2p_comm=True,
            )

        else:  # no p2p overlap
            output_tensor = forward_step_helper(
                forward_k, current_microbatch, checkpoint_activations_microbatch
            )

            # Backward pass.
            backward_k = k
            input_tensor_grad = backward_step_helper(backward_k)

            # Send output_tensor and input_tensor_grad, receive input_tensor
            # and output_tensor_grad.

            # Determine if current stage has anything to send in either direction,
            # otherwise set tensor to None.
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            parallel_state.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
            if parallel_state.is_pipeline_last_stage():
                output_tensor = None

            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            parallel_state.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
            if parallel_state.is_pipeline_first_stage():
                input_tensor_grad = None

            # Determine if peers are sending, and where in data structure to put
            # received tensors.
            recv_prev = True
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                # First stage is ahead of last stage by (pipeline_parallel_size - 1).
                next_forward_model_chunk_id = get_model_chunk_id(
                    forward_k - (pipeline_parallel_size - 1), forward=True
                )
                if next_forward_model_chunk_id == (num_model_chunks - 1):
                    recv_prev = False
                next_forward_model_chunk_id += 1
            else:
                next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1, forward=True)

            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
                next_backward_model_chunk_id = get_model_chunk_id(
                    backward_k - (pipeline_parallel_size - 1), forward=False
                )
                if next_backward_model_chunk_id == 0:
                    recv_next = False
                next_backward_model_chunk_id -= 1
            else:
                next_backward_model_chunk_id = get_model_chunk_id(backward_k + 1, forward=False)

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.
            if k == (num_microbatches_remaining - 1):
                recv_prev = False

            # Communicate tensors.
            (
                input_tensor,
                output_tensor_grad,
            ) = p2p_communication.send_forward_backward_recv_forward_backward(
                output_tensor,
                input_tensor_grad,
                recv_prev=recv_prev,
                recv_next=recv_next,
                tensor_shape=tensor_shape,
                config=config,
            )
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

        # Put input_tensor and output_tensor_grad in data structures in the
        # right location.
        if recv_prev:
            input_tensors[next_forward_model_chunk_id].append(input_tensor)
        if recv_next:
            output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    # Run cooldown backward passes (flush out pipeline).
    if not forward_only:
        if config.overlap_p2p_comm and bwd_wait_handles is not None:
            for wait_handle in bwd_wait_handles:
                wait_handle.wait()

        if all_warmup_microbatches:
            output_tensor_grads[num_model_chunks - 1].append(
                p2p_communication.recv_backward(tensor_shape, config=config)
            )
        for k in range(num_microbatches_remaining, total_num_microbatches):
            input_tensor_grad = backward_step_helper(k)
            next_backward_model_chunk_id = get_model_chunk_id(k + 1, forward=False)
            recv_next = True
            if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                if next_backward_model_chunk_id == (num_model_chunks - 1):
                    recv_next = False
            if k == (total_num_microbatches - 1):
                recv_next = False
            output_tensor_grads[next_backward_model_chunk_id].append(
                p2p_communication.send_backward_recv_backward(
                    input_tensor_grad, recv_next=recv_next, tensor_shape=tensor_shape, config=config
                )
            )

        # Launch any remaining grad reductions.
        enable_grad_sync()
        if config.grad_sync_func is not None:
            for model_chunk_id in range(num_model_chunks):
                if model_chunk_id not in synchronized_model_chunks:
                    config.grad_sync_func[model_chunk_id](model[model_chunk_id].parameters())
                    synchronized_model_chunks.add(model_chunk_id)

    if config.finalize_model_grads_func is not None and not forward_only:
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func(
            model, total_num_tokens if config.calculate_per_token_loss else None
        )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    return forward_data_store


def get_tensor_shapes(
    *,
    rank: int,
    model_type: ModelType,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int,
    config,
):
    # Determine right tensor sizes (based on position of rank with respect to split
    # rank) and model size.
    # Send two tensors if model is T5 and rank is in decoder stage:
    #     first tensor is decoder (pre-transpose),
    #     second tensor is encoder (post-transpose).
    # If model is T5 and rank is at the boundary:
    #     send one tensor (post-transpose from encoder).
    # Otherwise, send one tensor (pre-transpose).
    tensor_shapes = []

    seq_length = seq_length // parallel_state.get_context_parallel_world_size()
    if model_type == ModelType.encoder_and_decoder:
        decoder_seq_length = decoder_seq_length // parallel_state.get_context_parallel_world_size()

    if config.sequence_parallel:
        seq_length = seq_length // parallel_state.get_tensor_model_parallel_world_size()
        if model_type == ModelType.encoder_and_decoder:
            decoder_seq_length = (
                decoder_seq_length // parallel_state.get_tensor_model_parallel_world_size()
            )

    if model_type == ModelType.encoder_and_decoder:
        if parallel_state.is_pipeline_stage_before_split(rank):
            tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
        else:
            tensor_shapes.append((decoder_seq_length, micro_batch_size, config.hidden_size))
            tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
    else:
        tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
    return tensor_shapes


def recv_forward(tensor_shapes, config):
    input_tensors = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            input_tensors.append(None)
        else:
            input_tensors.append(p2p_communication.recv_forward(tensor_shape, config))
    return input_tensors


def recv_backward(tensor_shapes, config):
    output_tensor_grads = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            output_tensor_grads.append(None)
        else:
            output_tensor_grads.append(p2p_communication.recv_backward(tensor_shape, config))
    return output_tensor_grads


def send_forward(output_tensors, tensor_shapes, config):
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    for (output_tensor, tensor_shape) in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_forward(output_tensor, config)


def send_backward(input_tensor_grads, tensor_shapes, config):
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    for (input_tensor_grad, tensor_shape) in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_backward(input_tensor_grad, config)


def send_forward_recv_backward(output_tensors, tensor_shapes, config):
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    output_tensor_grads = []
    for (output_tensor, tensor_shape) in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            output_tensor_grads.append(None)
            continue
        output_tensor_grad = p2p_communication.send_forward_recv_backward(
            output_tensor, tensor_shape, config
        )
        output_tensor_grads.append(output_tensor_grad)
    return output_tensor_grads


def send_backward_recv_forward(input_tensor_grads, tensor_shapes, config):
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    input_tensors = []
    for (input_tensor_grad, tensor_shape) in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            input_tensors.append(None)
            continue
        input_tensor = p2p_communication.send_backward_recv_forward(
            input_tensor_grad, tensor_shape, config
        )
        input_tensors.append(input_tensor)
    return input_tensors


def forward_backward_pipelining_without_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages.

    Returns dictionary with losses if the last stage, empty dict otherwise."""

    if isinstance(model, list):
        assert (
            len(model) == 1
        ), "non-interleaved pipeline parallelism does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.overlap_p2p_comm:
        raise ValueError(
            "Non-interleaved pipeline parallelism does not support overlapping p2p communication"
        )

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Compute number of warmup microbatches.
    num_warmup_microbatches = (
        parallel_state.get_pipeline_model_parallel_world_size()
        - parallel_state.get_pipeline_model_parallel_rank()
        - 1
    )
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    model_type = get_model_type(model)

    rank = parallel_state.get_pipeline_model_parallel_rank()
    recv_tensor_shapes = get_tensor_shapes(
        rank=rank - 1,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
    )
    send_tensor_shapes = get_tensor_shapes(
        rank=rank,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
    )

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()

    if not forward_only:
        input_tensors = []
        output_tensors = []
    forward_data_store = []

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                i % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        input_tensor = recv_forward(recv_tensor_shapes, config)
        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(first_val_step, forward_only, i == 0),
            current_microbatch=i,
        )
        send_forward(output_tensor, send_tensor_shapes, config)
        total_num_tokens += num_tokens.item()

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        input_tensor = recv_forward(recv_tensor_shapes, config)

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = i == (num_microbatches_remaining - 1)

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                (i + num_warmup_microbatches) % max_outstanding_backprops
            ) >= config.num_microbatches_with_partial_activation_checkpoints
        else:
            checkpoint_activations_microbatch = None

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(
                first_val_step, forward_only, (i == 0) and (num_warmup_microbatches == 0)
            ),
            current_microbatch=i + num_warmup_microbatches,
        )
        total_num_tokens += num_tokens.item()

        if forward_only:
            send_forward(output_tensor, send_tensor_shapes, config)

            if not last_iteration:
                input_tensor = recv_forward(recv_tensor_shapes, config)

        else:
            output_tensor_grad = send_forward_recv_backward(
                output_tensor, send_tensor_shapes, config
            )

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            # Enable grad sync for the last microbatch in the batch if the full
            # backward pass completes in the 1F1B stage.
            if num_warmup_microbatches == 0 and last_iteration:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )

            if last_iteration:
                input_tensor = None
                send_backward(input_tensor_grad, recv_tensor_shapes, config)
            else:
                input_tensor = send_backward_recv_forward(
                    input_tensor_grad, recv_tensor_shapes, config
                )

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):

            # Enable async grad reduction in the last backward pass
            # Note: If grad sync function is provided, only enable
            # async grad reduction in first pipeline stage. Other
            # pipeline stages do grad reduction during pipeline
            # bubble.
            if i == num_warmup_microbatches - 1:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = recv_backward(send_tensor_shapes, config)

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )

            send_backward(input_tensor_grad, recv_tensor_shapes, config)

        # Launch any remaining grad reductions.
        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())

    if config.finalize_model_grads_func is not None and not forward_only:
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func(
            [model], total_num_tokens if config.calculate_per_token_loss else None
        )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    return forward_data_store
