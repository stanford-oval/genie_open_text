try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import os.path
import pandas as pd
import argparse
from helper_func import sprint
from tqdm import tqdm


def bounty_processing(input_file):
    d = {"PostId": [], "BountyAmount": []}

    tag_count = sum(1 for event, _ in ET.iterparse(input_file) if event == "end")

    with tqdm(total=tag_count, desc="Processing XML") as pbar:
        for event, elem in ET.iterparse(input_file):
            if event == "end":
                try:
                    if "BountyAmount" in elem.attrib:
                        postid = int(elem.attrib["PostId"])
                        bounty = int(elem.attrib["BountyAmount"])
                        d["PostId"].append(postid)
                        d["BountyAmount"].append(bounty)
                    # print elem.tag,elem.attrib
                    elem.clear()
                except Exception as e:
                    pass
                pbar.update(1)

    answerid_questionid_file = os.path.join(os.path.dirname(input_file), "AnswerId_QuestionId.csv")
    answerid_questionid = pd.read_csv(answerid_questionid_file)

    question_bounty = {"QuestionId": [], "Bounty": []}
    for postid, bounty in zip(d["PostId"], d["BountyAmount"]):
        if answerid_questionid[answerid_questionid["QuestionId"] == postid].index.tolist():
            question_bounty["QuestionId"].append(postid)
            question_bounty["Bounty"].append(bounty)

    file_dir = os.path.dirname(os.path.abspath(input_file))
    output_file = os.path.join(file_dir, "QuestionId_Bounty.csv")
    df = pd.DataFrame(question_bounty)
    df.to_csv(output_file, index=False, columns=["QuestionId", "Bounty"])
    print("***********************************")
    print("output file: %s" % output_file)
    sprint(file_dir, "pystack_analysis.log", "# questions having bounty: %d" % len(df))
    print("***********************************")


if __name__ == "__main__":
    '''
    process */Votes.xml
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="../dataset/ai/Votes.xml",
                        help="input: */Votes.xml, output: */QuestionId_Bounty.csv")
    args = parser.parse_args()
    input_file = args.input
    print("input file %s " % input_file)
    bounty_processing(input_file)
