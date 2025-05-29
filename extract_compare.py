import json

# Step 1: Load the two JSON files
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Load both files
file_a = load_json('/home/jfxiao/yunfeixie/MM-EUREKA_new/MathVerse_testmini_MM-EUREKA_z3_rloo_qwenvl2_5_lambd_3d_snake.sh_3d_orient_snake_many_ckpt_global_step295_hf_250508123806_original_extract.json')
file_b = load_json('/home/jfxiao/yunfeixie/MM-EUREKA_new/MathVerse_testmini_Qwen2.5-VL-7B-Instruct_250315070552_extract_gemini-2.0-flash.json')

# Step 2: Find items where score is true in A but false in B
result_a = {}
result_b = {}

for sample_index in file_a:
    if sample_index in file_b:
        if file_a[sample_index].get('score') is True and file_b[sample_index].get('score') is False:
            # Store the item from file A
            result_a[sample_index] = file_a[sample_index]
            # Store the corresponding item from file B
            result_b[sample_index] = file_b[sample_index]

# Step 3: Write the results to new JSON files
with open('/home/jfxiao/yunfeixie/MM-EUREKA_new/compare_result_file_a.json', 'w') as f:
    json.dump(result_a, f, indent=4)

with open('/home/jfxiao/yunfeixie/MM-EUREKA_new/compare_result_file_b.json', 'w') as f:
    json.dump(result_b, f, indent=4)

print(f"Found {len(result_a)} items where score is true in A but false in B.")
print(f"Results saved to compare_result_file_a.json and compare_result_file_b.json")