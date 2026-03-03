import json

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    # Ensure everything is a list of objects
    return data if isinstance(data, list) else [data]

def merge_and_sort_json(file1, file2, output_file):
    data1 = load_json(file1)
    data2 = load_json(file2)

    combined = data1 + data2

    # Deduplicate based on (P, M, K, N, L, C)
    unique_dict = {}
    for item in combined:
        # Default C = 1 if not given
        if "C" not in item:
            mean_cycle = item.pop("mean_cycle", None)
            item["C"] = 1
            item["mean_cycle"] = mean_cycle
        key = (
            item.get("P"),
            item.get("M"),
            item.get("K"),
            item.get("N"),
            item.get("L"),
            item.get("C"),
        )
        unique_dict[key] = item  # later ones override earlier ones

    # Convert back to list
    merged_list = list(unique_dict.values())

    # Sort by numeric keys (P, M, K, N, L, C)
    merged_list.sort(key=lambda x: tuple(x.get(k, 1) for k in ["P", "M", "K", "N", "L", "C"]))

    # Save result
    with open(output_file, "w") as f:
        json.dump(merged_list, f, indent=2)

    print(f"Merged and sorted JSON written to {output_file}")

# Example usage:
merge_and_sort_json("async_sim_results.json", "col-major-gemm.json", "sim_results.json")
