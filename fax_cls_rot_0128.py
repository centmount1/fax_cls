# -*- coding: utf-8 -*-

# ライブラリのインポート
import os
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models, transforms
import numpy as np
import datetime
import random
import streamlit as st
import shutil
import fitz
import io
from PIL import Image
import glob
import sys

"""
FAX分類と回転処理
ターミナルでstreamlit run fax_cls_rot_0128.pyで実行
1. dataフォルダからPDFファイルを取得(空になっているのでpdfファイルのサンプルを入れて下さい)
2. new_dataフォルダにコピー
3. new_dataフォルダのPDFファイルを取得
4. モデルをロード
5. テストデータの予測
6. 分類されたファイルを回転処理して、news_listフォルダに移動
7. その他のファイルをothersフォルダに移動
"""


def main():
    # streamlitでの表示
    st.title('Classify NEWS_LIST or OTHERS')
    # ファイルディレクトリ
    FILE_DIR = Path(__file__).parent.absolute()
    # 今日と昨日の日付取得
    TODAY = datetime.date.today().strftime('%Y%m%d')
    YESTERDAY = str(int(TODAY)-1)

    # 乱数の固定
    def fix_seed(seed=1234):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    fix_seed(seed=42)

    # データ取得先
    # テスト用にdataフォルダを指定
    DIR_PATH = f'{FILE_DIR}/data/'

    # ファイル保存先フォルダの作成
    if not os.path.exists(f'{FILE_DIR}/new_data/'):
        os.mkdir(f'{FILE_DIR}/new_data/')

    # テスト用にdataフォルダのファイルを全て取得
    files = [x for x in os.listdir(DIR_PATH)]

    # 昨日と今日のファイルのリストを取得する場合（取得頻度によって調整必要です）
    # files = [x for x in os.listdir(DIR_PATH) if (f"{TODAY}" in x) or (f"{YESTERDAY}" in x)]

    # streamlitの表示
    progress_text1 = "ファイルのコピー中です"
    # プログレスバー
    my_bar1 = st.progress(0, text=progress_text1)
    # ファイルコピー
    for i, file in enumerate(files):
        my_bar1.progress((i + 1) / len(files), text=progress_text1)
        try:
            shutil.copy(f'{DIR_PATH}/{file}', f'{FILE_DIR}/new_data/{file}')
        except FileNotFoundError:
            pass
        except OSError:
            pass
    # プログレスバーを空にする
    my_bar1.empty()

    # new_dataフォルダのファイルリスト取得＆表示
    if os.path.exists(f'{FILE_DIR}/new_data/'):
        # PDFファイルのリストを取得
        test_data = [x for x in os.listdir(f'{FILE_DIR}/new_data/') if ".pdf" in x]
        st.write('PDFファイルのリスト')
        # コピーして取得したファイルを一覧表示
        df_test = pd.DataFrame(test_data, columns=['filename'])
        st.write(df_test)

        # データセット作成
        class Test_Datasets(Dataset):
            def __init__(self, data_transform, df):
                super().__init__()
                self.df = df
                self.data_transform = data_transform

            def __len__(self):
                return len(self.df)

            def __getitem__(self, index):
                file = self.df['filename'][index]
                # print(DIR_PATH + file)
                images = fitz.open(f'{FILE_DIR}/new_data/{file}')
                # 最初のページを取得
                page = images[0]
                # pixmapを取得
                matrix = fitz.Matrix(1, 1)
                pixmap = page.get_pixmap(matrix=matrix)
                # pillowでpng形式のバイトオブジェクトを取得
                byte = pixmap.pil_tobytes("png")
                # PIL画像を取得
                binary = io.BytesIO(byte)
                pil_img = Image.open(binary)
                image = self.data_transform(pil_img)
                return image, file

        # データの前処理transforms作成
        transform_test = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])

        # datasetのインスタンス作成, 前処理
        test_data = Test_Datasets(data_transform=transform_test, df=df_test)

        # データローダー作成
        dataloader_test = torch.utils.data.DataLoader(
            test_data,
            batch_size=1,
            shuffle=False
        )

        # CPUを利用
        DEVICE = torch.device("cpu")        

        # モデルの定義
        def get_model(target_num, isPretrained=False):
            if (isPretrained):
                weights = models.EfficientNet_V2_S_Weights.DEFAULT
                model_ft = models.efficientnet_v2_s(weights=weights)
            else:
                model_ft = models.efficientnet_v2_s()

            model_ft.classifier[1] = nn.Linear(1280, target_num)
            model_ft = model_ft.to(DEVICE)
            return model_ft

        # モデル設定（分類モデル・回転モデル）
        model_cls_ft = get_model(target_num=2, isPretrained=True)
        model_rot_ft = get_model(target_num=4, isPretrained=True)

        # モデルをロード(分類モデル・回転モデル)
        # 分類モデル
        cls_model = model_cls_ft
        cls_model.load_state_dict(torch.load(f'{FILE_DIR}/effv2_cls2_2024-02-01_model.pth',
                                            map_location=lambda storage, loc: storage))
        cls_model = cls_model.to(DEVICE)

        # 回転モデル
        rot_model = model_rot_ft
        rot_model.load_state_dict(torch.load(f'{FILE_DIR}/effv2_rot_2024-02-01_model.pth',
                                            map_location=lambda storage, loc: storage))
        rot_model = rot_model.to(DEVICE)

        # テストデータの予測
        progress_text2 = "データの分類処理中です"
        # プログレスバー
        my_bar2 = st.progress(0, text=progress_text2)
        # 空のデータフレーム作成（分類、回転処理を記録するため）
        df_new = pd.DataFrame(columns=['file', 'pred', 'rot'],
                              index=[i for i in range(len(dataloader_test))])
        
        # 分類用フォルダ作成（ニュース項目等、その他に分類）
        if not os.path.exists(f'{FILE_DIR}/new_data/news_list'):
            os.mkdir(f'{FILE_DIR}/new_data/news_list')
        if not os.path.exists(f'{FILE_DIR}/new_data/others'):
            os.mkdir(f'{FILE_DIR}/new_data/others')

        # 分類処理を実行
        for i, (input, file) in enumerate(dataloader_test):
            my_bar2.progress((i + 1) / len(dataloader_test), text=progress_text2)
            try:
                input = input.to(DEVICE)
                cls_model.eval()
                output = cls_model(input)
                # print(output)
                _, pred = torch.max(output, dim=1)
                df_new.iloc[i, 0] = file[0]
                df_new.iloc[i, 1] = pred.item()
                # ニュース項目等と分類された場合
                if pred.item() == 0:
                    # 回転処理の実行
                    rot_model.eval()
                    output2 = rot_model(input)
                    _, pred2 = torch.max(output2, dim=1)
                    df_new.iloc[i, 2] = pred2.item()
                    # 上向きの場合、回転しない
                    if pred2.item() == 0:
                        pass
                    # 下向きの場合、180度回転
                    if pred2.item() == 1:
                        doc = fitz.open(f'{FILE_DIR}/new_data/{file[0]}')
                        for page in doc:
                            page.set_rotation(180)
                            doc.save(f'{FILE_DIR}/new_data/{file[0]}',
                                     incremental=True, encryption=0)
                        doc.close()
                    # 左向きの場合、90度回転
                    elif pred2.item() == 2:
                        doc = fitz.open(f'{FILE_DIR}/new_data/{file[0]}')
                        for page in doc:
                            page.set_rotation(90)
                            doc.save(f'{FILE_DIR}/new_data/{file[0]}',
                                     incremental=True, encryption=0)
                        doc.close()
                    # 右向きの場合、270度回転
                    elif pred2.item() == 3:
                        doc = fitz.open(f'{FILE_DIR}/new_data/{file[0]}')
                        for page in doc:
                            page.set_rotation(270)
                            doc.save(f'{FILE_DIR}/new_data/{file[0]}',
                                     incremental=True, encryption=0)
                        doc.close()
                    # news_listフォルダに移動
                    shutil.move(f'{FILE_DIR}/new_data/{file[0]}',
                                f'{FILE_DIR}/new_data/news_list/{file[0]}',
                                copy_function=shutil.copy2)

                else:
                    # ニュース項目等に分類されない場合、othersフォルダに移動
                    shutil.move(f'{FILE_DIR}/new_data/{file[0]}',
                                f'{FILE_DIR}/new_data/others/{file[0]}',
                                copy_function=shutil.copy2)
            except Exception as e:
                print(e)

        # 回転処理後、new_dataフォルダにnews_listのファイルが残る場合がある
        # 移動先news_listフォルダに同じファイルがあれば削除。なければ移動。
        for p in glob.glob(f'{FILE_DIR}/new_data/*.pdf'):
            if os.path.exists(f'{FILE_DIR}/new_data/news_list/{os.path.basename(p)}'):
                os.remove(f'{FILE_DIR}/new_data/news_list/{os.path.basename(p)}')
            shutil.move(p, f'{FILE_DIR}/new_data/news_list',
                        copy_function=shutil.copy2)
        # プログレスバーを空にする
        my_bar2.empty()
        st.write('処理が終了しました')


if __name__ == '__main__':
    main()
    sys.exit()
