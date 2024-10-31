# convert_dataset.py  
主に強化学習用のデータセットを作るためのpythonプロジェクト。 
***
### class netwrk_test()
networkXを使ったノード配置の図示を試すためのもの。
#
### convert_exist2pos()
<details><summary>引数・返値詳細</summary>

##### 引数：( exsit_matrix , unit_h , unit_v )  
>**exist_matrix** : ノード位置を表した行列  
>**unit_h** : 水平方向の単位移動量  
>**unit_v** : 垂直方向の単位移動量  

##### 返り値 : (pos_mat , node_n)
>**pos_mat** : 各ノードのxy座標を格納した行列  
>**node_n** : ノードの総数

</details>

`0`(=false) または `1`(=true) で表された行列を座標空間に変換するための機能。  
行列の左下にあるノードが`(x,y)`=`(0,0)`、右に1マス進むと*x*座標が`unit_h`、上に1マス進むと*y*座標が`unit_v`だけ増加する。
#
### adj_node_tf()
<details><summary>引数・返値詳細</summary>

##### 引数：( n , max_n )
>**n** : 現在の座標位置  
>**max_n** : その方向における座標の最大値  
  
##### 返り値：( adj_range )
>**adj_range** : 隣接マスを表す配列  
</details>

`make_dist_mat()`の処理中に使用。これ単体で使うことはほぼない。  
#
### make_dist_mat()
<details><summary>引数・返値詳細</summary>

##### 引数：(exist_matrix , unit_h, unit_v )
>**exist_matrix** :   
>**unit_h** : 水平方向の単位移動量  
>**unit_v** : 垂直方向の単位移動量  
  
##### 返り値：(con_mat , 0 )
>**con_mat** (np.float32) : 各ノード同士の所要時間を格納した二次元配列  
>**0** (int) :`exsit_2_number()`での返り値の構造を合わせるために使用。
</details>

ワーシャルフロイド法に基づき、`exist_matrix`から各ノード間の移動時間の行列を生成する。
#
### get_tarain_dataset()
<details><summary>引数・返値詳細</summary>

##### 引数：( mode ) , 返り値 : ()
>**mode** : `'node_pos'` , `'disatance'` , `'conection'` , `'node_exsit'`のいずれか
- `'node_pos'`
  - ノード位置をxy座標で指定。
  - 返り値：
    - `rt_1` : 各ノードのxy座標
    - `rt_2` : ノードの総数
- `'distance'`
  - すべてのノード同士の移動時間を行列で指定。__(ノード数)__ 次元の正方行列で指定する必要がある。
  - 返り値：
    - `rt_1` : 各ノード同士の時間行列
    - `rt_2` : 0
- `'conection'`
  - ノード同士で直接繋がっている点の情報を格納。
  - 返り値：
    - `rt_1` : ノード同士の接続情報
    - `rt_2` : 0
- `'node_exist'`
  - N*Mマスの空間において、ノードが存在する場合は`1`、そうでない場合は`0`を指定。
  - 返り値：
    - `rt_1` : 各ノードのxy座標
    - `rt_2` : 0

</details>

引数`mode`に入れる文字列に応じてデータを返す。各種配列情報を直接弄りたいときに使用。  
#
### exsit_2_number()
<details><summary>引数・返値詳細</summary>

##### 引数：( mode )
>**mode** : `'node_exsit'` , `'dis'` のいずれか  
  
##### 返り値：( rt_data )
>**rt_data** (tuple) : `mode`に応じたタプル。  
</details>
  
`get_tarain_dataset('node_exsit')`において`0`,`1`で表されたノード情報から*xy座標*または*各ノード間の所要時間*を返す機能。
#
### 基本的な使い方
`get_tarain_dataset('node_exsit')`で空間情報を書き込み、それを`exsit_2_number()`で呼び出す。  
***
# 更新情報
- **2024.10.28** 
  - ローカルでこねこねしていた分をリポジトリに追加
- **2024.10.29** 
  - `make_dist_mat()`にて点が存在しない場所の情報を保持してしまう問題の修正
  - `TSNE`の参照がごく一部だったため `t_sne.py`の内容を`train_test_detaset.py`に統合
- **2024.10.31**
  - `train_test_dataset.py`の`train()`にて強化学習の計算が遅かった問題を改善
    - distanceの行列を算出するたびに毎回`make_dist_mat()`が呼び出されていたのが原因
    - `test_dataset.py`内の`Class Random_Dataset()`に引数`dis_mtx`を追加し、distanceの行列を参照で渡せるように変更