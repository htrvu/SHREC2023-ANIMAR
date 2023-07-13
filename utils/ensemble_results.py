import os
import numpy as np
import pandas as pd
import json
import argparse

def save_submission(result_json_path, output_path):
    submit_dict={}
    with open(result_json_path) as f:
        result = json.load(f)
        for id in result:
            submit_dict[id]=result[id]['pred_ids']
        df=pd.DataFrame.from_dict(submit_dict).T
        df.to_csv(output_path,sep=',',index=True,header=None)


def ensemble_results(results_folders, output_folder): 
    '''Synthesize the scores by getting max value.

    Parameters
    ----------
    results_folders: 
        Paths to the folders contain query results.
        Example:
            `['predicts/pointmlp/pcl_predict_0', 'predicts/pointmlp/pcl_predict_1', ...]`
        Each folder has a json file `query_results.json` storing results. This file should be in format
            {
                <query_id>: {
                    "pred_ids": [<list of predicted ids>],
                    "scores": [<list of predicted ids' corresponding scores>]
                } 
            }
    output_folder:
        Path the output folder. This folder will store the synthetic result in JSON and CSV format.

    Returns
    -------
    None

    '''
    in_folders = os.listdir(results_folders)
    assert len(in_folders) >= 1, "There is no data passed in!" 

    print(f'Ensembling {len(in_folders)} query results...')

    json_paths = []
    for folder in in_folders:
        json_paths.append(os.path.join(results_folders, folder, 'query_results.json'))
    
    max_result = {}

    # Extract first file as a base to make comparison
    with open(json_paths[0]) as f:
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
    for path in json_paths[1:]:
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
    # folder_name = os.path.dirname(output_path)
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, 'query_results.json'), 'w') as f:
        json.dump(max_result, f, indent=2)
    save_submission(os.path.join(output_folder, 'query_results.json'), os.path.join(output_folder, 'submission.csv'))

    print(f'The results was saved in the folder {output_folder}')


#print(get_max_similarities.__doc__)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True, help='Path to the folder of query results to be ensembled')
    parser.add_argument('--output-folder', type=str, required=True, help='Path to the folder of query results to be ensembled')
    args = parser.parse_args()

    ensemble_results(args.input_folder, args.output_folder)

