def remap_state_dict_keys(state_dict):
    key_mapping = {}
    for original_key in state_dict.keys():
        key_parts = original_key.split(".")

        if key_parts[0] in ["block0", "block1", "block2", "block3", "block4"]:
            prefix = ["encoder_blocks"]
            key_parts[0] = key_parts[0][-1]
            key_parts[1] = "blocks"
            if key_parts[3] == "shortcut":
                key_parts[4] = "processing_block"
            else:
                key_parts[4] = "layers"
            out = prefix + key_parts

            key_mapping[original_key] = ".".join(out)
            continue

        if key_parts[0] in ["block5", "block6", "block7", "block8"]:
            prefix = ["decoder_blocks"]
            key_parts[0] = str(int(key_parts[0][-1]) - 5)
            key_parts[1] = "blocks"
            if key_parts[2] == "0":
                if key_parts[3] == "shortcut":
                    key_parts[4] = "processing_block"
                else:
                    key_parts[4] = "layers"
            else:
                if key_parts[3] != "shortcut":
                    key_parts[4] = "layers"
            out = prefix + key_parts

            key_mapping[original_key] = ".".join(out)
            continue

        if key_parts[0] == "conv15":
            key_parts[0] = "final_conv"
            key_parts[1] = "layers"
            out = key_parts

            key_mapping[original_key] = ".".join(out)
            continue

        if key_parts[0] == "block9":
            key_parts[0] = "final_block"
            key_parts[1] = "blocks"
            if key_parts[2] == "0":
                if key_parts[3] == "shortcut":
                    key_parts[4] = "processing_block"
                else:
                    key_parts[4] = "layers"
            else:
                if key_parts[3] != "shortcut":
                    key_parts[4] = "layers"
            out = key_parts

            key_mapping[original_key] = ".".join(out)
            continue

    for original_key, new_key in key_mapping.items():
        state_dict[new_key] = state_dict.pop(original_key)

    return state_dict
