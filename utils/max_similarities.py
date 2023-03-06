import os
import numpy as np
import pandas as pd
import json

def get_max_similarities(results_paths, output_path): 
    '''Synthesize the scores by getting max value.

    Parameters
    ----------
    results_paths: 
        Paths to the json file storing results. It should be in format
            {
                <query_id>: {
                    "pred_ids": [<list of predicted ids>],
                    "scores": [<list of predicted ids' corresponding scores>]
                } 
            }
    output_paht:
        Path to store the synthetic result.

    Returns
    -------
    None

    '''

    assert isinstance(results_paths, (list, tuple)) and len(results_paths) > 1, "There is no data passed in!" 
    
    max_result = {}

    # Extract first file as a base to make comparison
    with open(results_paths[0]) as f:
        obj = json.load(f)
        # Turn {"pred_ids": [id1, id2], "scores": [s1, s2]} into
        #       {id1: s1, id2: s2}
        for query in obj:
            obj_query = obj[query]
            max_result[query] = {
                obj_query["pred_ids"][i]: obj_query["scores"][i] 
                for i in range(len(obj_query["pred_ids"]))
            } 
    
    # Compare with other results
    for path in results_paths[1:]:
        with open(path) as f:
            cur_result = json.load(f)
            for query in cur_result:
                cur_query_obj = cur_result[query]
                max_query_obj = max_result[query]

                assert len(cur_query_obj["pred_ids"]) == len(cur_query_obj["scores"]), "Lengths of predict ids and scores do not match!"

                # Get max of scores
                for i, id in enumerate(cur_query_obj["pred_ids"]):
                    max_query_obj[id] = max(max_query_obj[id], cur_query_obj["scores"][i])
    
    # Sort by scores and convert back to original structure
    for query in max_result:
        # Sort by scores
        sorted_preds = sorted(max_result[query].items(), key=lambda x:x[1], reverse = True)
        
        # Turn {id1: score1, id2: score2} back to 
        #       {pred_ids: [id1, id2], scores: [score1, score2]} 
        pred_ids = [pred[0] for pred in sorted_preds]
        scores = [pred[1] for pred in sorted_preds]
        max_result[query] = {
            "pred_ids": pred_ids,
            "scores": scores
        }

    # Export result to json
    folder_name = os.path.dirname(output_path)
    os.makedirs(folder_name, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(max_result, f, indent=2)

#print(get_max_similarities.__doc__)