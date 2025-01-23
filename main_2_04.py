#streamlit run main_2_04.py
import streamlit as st

import pandas as pd
import numpy as np

import re
import json
import os

import torch
import torch.nn.functional as F
from transformers import BertJapaneseTokenizer, BertModel


#センテンスバートの内容ｰｰｰｰｰｰ
class SentenceBertJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest",
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        # return torch.stack(all_embeddings).numpy()
        return torch.stack(all_embeddings)

#モデルの指定
MODEL_NAME = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"  # <- v2です。
model = SentenceBertJapanese(MODEL_NAME)

##ここまでモデル----------
# df読み込み
df = pd.read_csv('m2022.csv', header=0)

# vec読み込み
with open('m2022.json', 'r') as f:
    loaded_vecs_json = json.load(f)
    loaded_vecs = np.array(loaded_vecs_json)  # NumPy配列に変換

##ここまで読み込み----------

# タイトルとテキストを記入
st.title('問題から検索版')

# ページ上部へのリンク
st.markdown("<a name='top'></a>", unsafe_allow_html=True)

# 重複する値を一つにまとめる
unique_years = df['Year'].drop_duplicates()
unique_categories = df['Category'].drop_duplicates()
unique_subtitles = df['Subtitle'].drop_duplicates()

#選択してもらうとこ
selected_category = st.selectbox('カテゴリ', unique_categories)
selected_year = st.selectbox('年度', unique_years)
selected_subtitle = st.selectbox('問題番号', unique_subtitles)

# 選択されたカテゴリ、年度、番号に基づいてデータフレームをフィルタリング
filtered_df = df[(df['Year'] == selected_year) & (df['Category'] == selected_category) & (df['Subtitle'] == selected_subtitle)]

# Title列を除外して表示
if 'Title' in filtered_df.columns:
    filtered_df_tno = filtered_df.drop(columns=['Title'])


# バッチサイズを設定
batch_size = 100

# mondai_vecsの準備
mondai_vecs = torch.tensor(loaded_vecs[filtered_df.index])

# バッチごとのコサイン類似度を保存するためのリスト
similarity_scores = []

# バッチ処理
for i in range(0, len(loaded_vecs), batch_size):
    # バッチを作成
    batch = torch.tensor(loaded_vecs[i : i + batch_size])
    
    # バッチとmondai_vecsのコサイン類似度を計算
    sim = F.cosine_similarity(mondai_vecs, batch)
    
    # スコアを保存
    similarity_scores.append(sim.tolist())

# 全てのバッチを通じたコサイン類似度のリストを平坦化
similarity_scores = [score for sublist in similarity_scores for score in sublist]

df_results = pd.DataFrame(
    
 {
    '類似度': sim,
    'カテゴリ': df['Category'],
    '年度': df['Year'].astype(str) + "年",
    '番号': df['Subtitle'],
    '問題文': df['Questions'],
}

)

df_results = df_results.sort_values(by='類似度', ascending=False)

st.write('類似度の高い問題を10こ表示させています：')
st.dataframe(df_results.head(10))


# 区切り線を表示
st.markdown("---")
st.markdown("## 一番似ている問題です。解いてくだちぃ")
# st.dataframe(df_results.iloc[1])





text = df_results['問題文'].iloc[1]

# 括弧内のピリオドを一時的に別の文字列に置換
def replacer(match):
    return match.group(0).replace('。', '<period>')

text = re.sub(r'\（.*?\）', replacer, text)

# 置換されたテキストをピリオドで分割
sentences = text.split('。')

# 分割した各文について、一時的に置換した括弧内のピリオドを元に戻す
sentences = [sentence.replace('<period>', '。') for sentence in sentences]

# ピリオドを再追加して各文を結合し、表示
st.markdown('\n\n'.join(sentence + '。' for sentence in sentences if sentence))



st.markdown("---")





df_ans = pd.read_csv('m2022_ans.csv', header=0)

df_results_ans = pd.DataFrame(
    {
        '類似度': sim,
        'カテゴリ': df_ans['Category'],
        '年度': df_ans['Year'].astype(str) + "年",
        '番号': df_ans['Subtitle'],
        '問題文': df_ans['Questions'],
        'Answer': df_ans['Answer'], 
    }
)

df_results_ans = df_results_ans.sort_values(by='類似度', ascending=False)



# ボタンを作成
option = st.radio(
    "正解の選択肢を選んでください：",
    ("選択肢1", "選択肢2", "選択肢3", "選択肢4")
)

# 選択肢の番号とその対応する値の辞書を作成
option_dict = {"選択肢1": "1", "選択肢2": "2", "選択肢3": "3", "選択肢4": "4"}

# 選択された選択肢の値を取得
selected_option_value = option_dict[option]

# 選択された選択肢とdf['Answer']の値を比較
if selected_option_value == df_results_ans['Answer'].iloc[1]:
    result = "正解"
else:
    result = "不正解"

# st.write(f"選択されたオプション：{option}")
st.write(f"結果：{result}")

st.markdown("---")



st.write("下のボタンを押すと解答が表示されるよ！複数回答とかは正解の選択肢がないよ！")
st.write("まさかの手打ちなので間違ってるかもだよ！解答は自分でもしらべてね！")

# ボタンが押されたらデータフレームを表示する
if st.button('解答部分を表示'):
    st.dataframe(df_results_ans.iloc[1])




# ページ上部へのリンク
st.markdown("<a name='top'></a>", unsafe_allow_html=True)





#ーーーーーーーーここから保存

st.markdown("---")


# ユーザーの評価を受け取る
evaluation = st.radio(
    "評価を選択してください:",
    ("参考になった", "参考にならなかった")
)

# ユーザー名の入力を受け取る
username = st.text_input("ユーザー名を入力してください:")

# ユーザーの入力を受け取る
user_input = st.text_area("参考になった理由を教えてください:")

# 類似度が高い問題のタイトルを取得
top_result_index = df_results.index[1]  # 2番目に類似度が高い問題のインデックスを取得
top_result_title = df.loc[top_result_index, 'Title']  # インデックスを使ってタイトルを取得

if st.button("保存"):
    selected_question_title = filtered_df['Title'].values[0]  # 現在選択されている問題のタイトルを取得
    new_data = pd.DataFrame({
        'ユーザー名': [username],
        '入力テキスト': [user_input],
        '評価': [evaluation],
        '類似度が高い問題のタイトル': [top_result_title],  # 最も類似度が高い問題のタイトルを保存
        '選択した問題のタイトル': [selected_question_title],  # 現在選択している問題のタイトルを保存
    })

    if os.path.isfile('user_input.csv'):
        # 既存のデータを読み込み
        df = pd.read_csv('user_input.csv')
        # 新規データを追加
        df = df.append(new_data, ignore_index=True)
    else:
        df = new_data

    # DataFrameをCSVファイルに保存
    df.to_csv('user_input.csv', index=False)
    st.success("テキストが保存されました。")

# 画面下部にある「上に戻る」ボタン
if st.button('一番上に戻る'):
    st.markdown("<a href='#top'>Go to top</a>", unsafe_allow_html=True)

st.markdown("---")

# ユーザー名でデータを検索
search_username = st.text_input("表示するユーザー名を入力してください:")

if st.button("検索"):
    if os.path.isfile('user_input.csv'):
        df = pd.read_csv('user_input.csv')
        df = df[df['ユーザー名'] == search_username]  # ユーザー名が一致する行だけを選択
        st.write(df)  # DataFrameを表示
    else:
        st.error("まだデータがありません。")