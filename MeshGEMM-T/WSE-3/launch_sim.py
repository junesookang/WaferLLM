import random
import numpy as np
import math
import argparse
import struct
import json
from pathlib import Path

from cerebras.sdk.sdk_utils import input_array_to_u32, memcpy_view  # type: ignore
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime  # type: ignore
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType, MemcpyOrder  # type: ignore

def parse_args():
    #[N,K]@[K,M]
    parser = argparse.ArgumentParser(description="MeshGEMM on simulator")
    parser.add_argument("--P", required=True, type=int, help="PEs rectangle size: P x P")
    parser.add_argument("--M", required=True, type=int, help="Input context length")
    parser.add_argument("--K", required=True, type=int, help="Word vector dimension")
    parser.add_argument("--N", required=True, type=int, help="Output dimension")
    parser.add_argument("--L", required=False, type=int, default=1, help="Computation Loop to help Benchmarking")

    args = parser.parse_args()
    return args

def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])

def make_u48(words):
    return words[0] + (words[1] << 16) + (words[2] << 32)

def assignId(pc, P):
    send_id = 0
    recv_id = 0

    pc = pc + 1

    if pc%2 == 0:
        send_id = pc - 2
        recv_id = pc + 2
    else:
        send_id = pc + 2
        recv_id = pc - 2

    if pc == 1:
        send_id = 3
        recv_id = 2

    if pc == 2:
        send_id = 1
        recv_id = min(recv_id, P)

    if P%2 == 0:
        if pc == P-1:
            send_id = P
            recv_id = P - 3
        if pc == P:
            send_id = P - 2
            recv_id = P - 1
    else:
        if pc == P-1:
            send_id = max(send_id, 1)
            recv_id = P
        if pc == P:
            send_id = P - 1
            recv_id = P - 2
    return send_id - 1, recv_id - 1


def main():
    random.seed(2025)

    args = parse_args()

    P = args.P

    orig_M = args.M
    orig_K = args.K
    orig_N = args.N

    M = orig_M
    K = orig_K
    N = orig_N

    L = args.L

    Mt = math.ceil(M / P)
    Kt = math.ceil(K / P)
    Nt = math.ceil(N / P)

    M = Mt * P
    K = Kt * P
    N = Nt * P

    file_name = "sim_results.json"
    file_path = Path(file_name)
    existing_results = []

    if file_path.exists():
        try:
            file_content = file_path.read_text().strip()
            if file_content:
                parsed = json.loads(file_content)
                if isinstance(parsed, list):
                    existing_results = parsed
        except json.JSONDecodeError:
            existing_results = []

    if any(
        entry.get("P") == P
        and entry.get("M") == M
        and entry.get("K") == K
        and entry.get("N") == N
        and entry.get("L") == L
        for entry in existing_results
    ):
        print(f"Result for P={P}, M={M}, K={K}, N={N}, L={L} already exists. Skipping simulation.")
        return

    io_dtype = MemcpyDataType.MEMCPY_16BIT
    memcpy_order = MemcpyOrder.ROW_MAJOR

    tensor_X = np.random.rand(M, K).astype(np.float16)

    tensor_W = np.random.rand(K, N).astype(np.float16)

    ind = np.zeros((P, P)).astype(int)

    for i in range(P):
        for j in range(P):
            if i == 0:
                ind[0, j] = j
            elif i == 1:
                _, ind[1, j] = assignId(ind[0, j], P)
            else:
                if (i-1)%2==0:
                    _, ind[i, j] = assignId(ind[i-2, j], P)
                else:
                    ind[i, j], _ = assignId(ind[i-2, j], P)

    tensor_W_offset = np.zeros((K, N)).astype(np.float16)

    for i in range(P):
        for j in range(P):
            t = ind[i, j]
            tensor_W_offset[i*Kt:(i+1)*Kt, j*Nt:(j+1)*Nt] = tensor_W[t*Kt:(t+1)*Kt, j*Nt:(j+1)*Nt]

    runner = SdkRuntime("out")
    runner.load()
    runner.run()

    sym_X = runner.get_id("X")
    sym_W = runner.get_id("W")

    sym_res = runner.get_id("res")

    symbol_time_memcpy = runner.get_id("time_memcpy")
    symbol_time_ref = runner.get_id("time_ref")

    X1 = tensor_X.reshape(P, Mt, P, Kt)
    X2 = X1.transpose(0, 2, 3, 1)
    X3 = X2.reshape(P, P, Mt*Kt)
    X_u32 = input_array_to_u32(X3.ravel(), 1, 1)
    runner.memcpy_h2d(sym_X, X_u32, 0, 0, P, P, Mt*Kt, \
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

    W1 = tensor_W_offset.reshape(P, Kt, P, Nt)
    W2 = W1.transpose(0, 2, 1, 3)
    W3 = W2.reshape(P, P, Kt*Nt)
    W_u32 = input_array_to_u32(W3.ravel(), 1, 1)
    runner.memcpy_h2d(sym_W, W_u32, 0, 0, P, P, Kt*Nt, \
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)

    runner.launch('init_task', nonblock=False)
    total_warmup_times, total_repeat_times = 0, 1
    runner.launch('meshgemm_host', np.int16(total_warmup_times), np.int16(total_repeat_times), nonblock=False)

    res3_1d_u32 = np.zeros(M*N, dtype=np.uint32)
    runner.memcpy_d2h(res3_1d_u32, sym_res, 0, 0, P, P, Mt*Nt, \
                      streaming=False, data_type=io_dtype, order=memcpy_order, nonblock=False)
    res3_1d_fp16 = memcpy_view(res3_1d_u32, np.dtype(np.float16))
    res3 = res3_1d_fp16.reshape((P, P, Nt, Mt))
    res2 = res3.transpose(0, 3, 1, 2)
    res = res2.reshape(M, N)

    runner.launch('init_task', nonblock=False)
    total_warmup_times, total_repeat_times = 1, 5
    runner.launch('meshgemm_host', np.int16(total_warmup_times), np.int16(total_repeat_times), nonblock=False)

    time_memcpy_1d_f32 = np.zeros(P*P*3, dtype=np.float32)
    runner.memcpy_d2h(time_memcpy_1d_f32, symbol_time_memcpy, 0, 0, P, P, 3, streaming=False,
                    order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
    time_memcpy_hwl = np.reshape(time_memcpy_1d_f32, (P, P, 3), order='C')

    time_ref_1d_f32 = np.zeros(P*P*2, np.float32)
    runner.memcpy_d2h(time_ref_1d_f32, symbol_time_ref, 0, 0, P, P, 2, streaming=False,
                    order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
    time_ref_hwl = np.reshape(time_ref_1d_f32, (P, P, 2), order='C')

    runner.stop()


    time_start = np.zeros((P, P)).astype(int)
    time_end = np.zeros((P, P)).astype(int)
    word = np.zeros(3).astype(np.uint16)
    for w in range(P):
        for h in range(P):
            hex_t0 = int(float_to_hex(time_memcpy_hwl[(h, w, 0)]), base=16)
            hex_t1 = int(float_to_hex(time_memcpy_hwl[(h, w, 1)]), base=16)
            hex_t2 = int(float_to_hex(time_memcpy_hwl[(h, w, 2)]), base=16)
            word[0] = hex_t0 & 0x0000ffff
            word[1] = (hex_t0 >> 16) & 0x0000ffff
            word[2] = hex_t1 & 0x0000ffff
            time_start[(h, w)] = make_u48(word)
            word[0] = (hex_t1 >> 16) & 0x0000ffff
            word[1] = hex_t2 & 0x0000ffff
            word[2] = (hex_t2 >> 16) & 0x0000ffff
            time_end[(h, w)] = make_u48(word)

    time_ref = np.zeros((P, P)).astype(int)
    word = np.zeros(3).astype(np.uint16)
    for w in range(P):
        for h in range(P):
            hex_t0 = int(float_to_hex(time_ref_hwl[(h, w, 0)]), base=16)
            hex_t1 = int(float_to_hex(time_ref_hwl[(h, w, 1)]), base=16)
            word[0] = hex_t0 & 0x0000ffff
            word[1] = (hex_t0 >> 16) & 0x0000ffff
            word[2] = hex_t1 & 0x0000ffff
            time_ref[(h, w)] = make_u48(word)

    for py in range(P):
        for px in range(P):
            time_ref[(py, px)] = time_ref[(py, px)] - (px + py)

    time_start = time_start - time_ref
    time_end = time_end - time_ref

    expected_res = np.matmul(tensor_X, tensor_W)

    #print("Expected result:")
    #print(expected_res)
    #print("Actual result:")
    #print(res)

    min_time_start = time_start.min()
    max_time_end = time_end.max()

    print(f"\nRepeat count: {total_repeat_times}")
    print(f"P: {P}, M: {M}, K: {K}, N: {N}, fmach computation loop: {L}")
    print(f"Mean cycle count: {np.mean(time_end - time_start)/total_repeat_times}")
    print(f"Max Cycle count: {(max_time_end - min_time_start)/total_repeat_times}")

    result_entry = {"P": P, "M": M, "K": K, "N": N, "L": L, "mean_cycle": np.mean(time_end - time_start)/total_repeat_times,}

    existing_results.append(result_entry)
    existing_results.sort(
        key=lambda entry: (
            entry.get("P", 0),
            entry.get("M", 0),
            entry.get("K", 0),
            entry.get("N", 0),
            entry.get("L", 0),
        )
    )
    with file_path.open("w") as f:
        json.dump(existing_results, f, indent=2)
        f.write("\n")

if __name__ == "__main__":
    main()
