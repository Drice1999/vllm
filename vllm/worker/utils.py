# SPDX-License-Identifier: Apache-2.0
'''
Worker-related helper functions.
'''

from vllm.utils import STR_NOT_IMPL_ENC_DEC_ERR_STRS
from vllm.worker.model_runner import GPUModelRunnerBase

import os
import psutil

from vllm.utils import import_pynvml
pynvml = import_pynvml()

def assert_enc_dec_mr_supported_scenario(
        enc_dec_mr: GPUModelRunnerBase) -> None:
    '''
    Asserted that the provided encoder/decoder model runner instance reflects
    a supported scenario.
    '''

    # Reminder: Please update docs/source/features/compatibility_matrix.md
    # If the feature combo become valid

    if enc_dec_mr.cache_config.enable_prefix_caching:
        raise NotImplementedError(
            STR_NOT_IMPL_ENC_DEC_ERR_STRS['STR_NOT_IMPL_ENC_DEC_PREFIX_CACHE'])

    if enc_dec_mr.sliding_window is not None:
        raise NotImplementedError(
            STR_NOT_IMPL_ENC_DEC_ERR_STRS['STR_NOT_IMPL_ENC_DEC_SWA'])

    if enc_dec_mr.scheduler_config.chunked_prefill_enabled:
        raise NotImplementedError(STR_NOT_IMPL_ENC_DEC_ERR_STRS[
            'STR_NOT_IMPL_ENC_DEC_CHUNKED_PREFILL'])

    if getattr(enc_dec_mr.model_config.hf_config, 'attn_logit_softcapping',
               None) is not None:
        raise NotImplementedError(
            STR_NOT_IMPL_ENC_DEC_ERR_STRS['STR_NOT_IMPL_ENC_DEC_LOGIT_SOFTCAP']
        )

    if enc_dec_mr.lora_config is not None:
        raise NotImplementedError(
            STR_NOT_IMPL_ENC_DEC_ERR_STRS['STR_NOT_IMPL_ENC_DEC_LORA'])

    if enc_dec_mr.parallel_config.pipeline_parallel_size > 1:
        raise NotImplementedError(
            STR_NOT_IMPL_ENC_DEC_ERR_STRS['STR_NOT_IMPL_ENC_DEC_PP'])

    if enc_dec_mr.scheduler_config.num_lookahead_slots > 0:
        raise NotImplementedError(
            STR_NOT_IMPL_ENC_DEC_ERR_STRS['STR_NOT_IMPL_ENC_DEC_SPEC_DEC'])

    if enc_dec_mr.prompt_adapter_config is not None:
        raise NotImplementedError(STR_NOT_IMPL_ENC_DEC_ERR_STRS[
            'STR_NOT_IMPL_ENC_DEC_PROMPT_ADAPTER'])

def get_numa_node_for_gpu(device_index):
    """
    Given a GPU device index (as visible to this process),
    returns its associated NUMA node by first querying its PCI info.
    """
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
    pci_bus_id = pci_info.busId  # e.g., "0000:81:00.0"
    
    numa_path = f"/sys/bus/pci/devices/{pci_bus_id}/numa_node"
    try:
        with open(numa_path, 'r') as f:
            numa_node = int(f.read().strip())
    except Exception as e:
        print(f"Error reading NUMA node for {pci_bus_id}: {e}")
        numa_node = 0  # default/fallback
    return numa_node

def get_visible_gpu_indices():
    """
    Returns a list of GPU indices based on CUDA_VISIBLE_DEVICES.
    If CUDA_VISIBLE_DEVICES is not set, return all available GPUs.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        # Assume comma-separated list of device indices (as strings)
        return [int(x) for x in visible.split(',') if x.strip() != '']
    else:
        count = pynvml.nvmlDeviceGetCount()
        return list(range(count))

def get_cpus_for_numa_node(numa_node):
    cpulist_file = f"/sys/devices/system/node/node{numa_node}/cpulist"
    with open(cpulist_file, 'r') as f:
        cpulist_str = f.read().strip()
    cpus = []
    for part in cpulist_str.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            cpus.extend(range(start, end + 1))
        else:
            cpus.append(int(part))
    return cpus

def pin_process_to_cpus():
    pynvml.nvmlInit()

    # Get GPU indices visible to your process.
    gpu_indices = get_visible_gpu_indices()

    if (len(gpu_indices) > 0):
        gpu_numa_node = 0
        has_multiple_numa_node = False

        for i in range(0, len(gpu_indices)):
            idx = gpu_indices[i]
            numa = get_numa_node_for_gpu(idx)
            if (i != 0 and gpu_numa_node != numa):
                has_multiple_numa_node = True
            gpu_numa_node = numa

        if (not has_multiple_numa_node):
            local_cpus = get_cpus_for_numa_node(gpu_numa_node)
        
        p = psutil.Process(os.getpid())
        p.cpu_affinity(local_cpus)

    pynvml.nvmlShutdown()