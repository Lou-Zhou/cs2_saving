#script which turns txt files of '/download/demo/{x}' to 'https://www.hltv.org/download/demo/{x}'
RAW_LINKS = "/home/lz80/cs2_saving/data_collection/raw_hltv_links" #input: raw txt file of /download/demo/{x}
HLTV_LINKS = "/home/lz80/cs2_saving/data_collection/hltv_links" #output

with open(RAW_LINKS, 'r') as file:
    content = file.read().split("\n")
    output_contents = [f'https://www.hltv.org{demo_end}' for demo_end in content]
    with open(HLTV_LINKS, "w") as text_file:
        text_file.write("\n".join(output_contents))