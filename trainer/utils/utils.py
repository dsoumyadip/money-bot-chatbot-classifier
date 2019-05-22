# Importing libraries
import re
import string
import pandas as pd


class Utils:
    CHAR_DICT = dict(zip([char for char in string.ascii_lowercase], range(26)))
    CHAR_DICT[' '] = 26

    def __init__(self):
        pass

    @staticmethod
    def get_data(path: str, **kwargs) -> pd.DataFrame:
        return pd.read_csv(path, **kwargs)

    @staticmethod
    def split_word(word: str) -> list:
        return [char for char in word]

    @staticmethod
    def clean_sentence(sentence: str) -> str:
        sentence = sentence.lower()
        match = re.compile('[^a-z\s]')
        return match.sub('', sentence)

    @classmethod
    def clean_external_data(cls, data: pd.DataFrame) -> pd.DataFrame:
        data2 = data[data['text'] != '[START]'].copy()
        data2['text'] = data2['text'].apply(lambda x: cls.clean_sentence(x))
        return data2.reset_index()

    @classmethod
    def make_conversation_pair(cls, data: pd.DataFrame) -> pd.DataFrame:
        external_chat_df = pd.DataFrame({"human": [], "robot": []})
        human_talks = ""
        robot_talks = ""

        for i in list(range(data.shape[0])):
            if data.source[i] == "human":
                human_talks = human_talks + data.text[i]
            else:
                robot_talks = robot_talks + data.text[i]

            if human_talks != "" and robot_talks != "":
                #                     print("Human talks: " + human_talks)
                #                     print("Robot talks: " + robot_talks)
                temp_df = pd.DataFrame({"human": [human_talks], "robot": [robot_talks]})
                external_chat_df = external_chat_df.append(temp_df)

                human_talks = ""
                robot_talks = ""

        return external_chat_df


def process_data(local_data_path: str, external_data_path: str, path_to_save: str) -> pd.DataFrame:
    print("Data processing is going on...")

    # Loading data from all sources
    local_data = Utils.get_data(local_data_path)
    external_data = Utils.get_data(external_data_path)

    local_data['CLIENT'] = local_data['CLIENT'].apply(lambda x: Utils.clean_sentence(x))
    local_data['CLIENT'] = local_data['CLIENT'].apply(lambda x: list(x))

    external_data['source'] = external_data['source'].apply(lambda x: Utils.clean_sentence(x))

    cleaned_external_data = Utils.clean_external_data(external_data)

    cleaned_external_data_pair = Utils.make_conversation_pair(cleaned_external_data)
    external_chat_df_filtered = cleaned_external_data_pair[cleaned_external_data_pair['human'].map(len) <= 30]

    external_chat_df_filtered = external_chat_df_filtered.rename(index=str,
                                                                 columns={"human": "CLIENT", "robot": 'ACTIVITY'})

    external_chat_df_filtered['CLIENT'] = external_chat_df_filtered['CLIENT'].apply(lambda x: x.lower())
    external_chat_df_filtered['CLIENT'] = external_chat_df_filtered['CLIENT'].apply(lambda x: list(x))

    all_df = local_data.append(external_chat_df_filtered).reset_index()
    all_df['CLIENT'] = all_df['CLIENT'].apply(lambda x: [Utils.CHAR_DICT[i] for i in x])

    return all_df





