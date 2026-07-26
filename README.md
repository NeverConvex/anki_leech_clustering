This project is meant to house modules that explore ways of clustering Anki (or other SRS program) leeches, primarily by grouping them according to measures of visual, phonetic, or semantic similarity. Using a combination of [approximate string matching](https://en.wikipedia.org/wiki/Approximate_string_matching) and various small-to-medium-scale neural networks (e.g., [manga-ocr](https://github.com/kha-white/manga-ocr) and [SmolVLM](https://huggingface.co/blog/smolvlm)) and various preexisting and custom clustering algorithms, I'm exploring a few different approaches for visual and phonetic similarity, but currently only include one of these in this public GitHub, which combines manga-ocr with the [default KMeans clustering algorithm available in sklearn](https://sklearn.org/stable/modules/generated/sklearn.cluster.KMeans.html). I expect to update this project in the future as I develop and test additional approaches.

The general idea behind most of these approaches is to use suitable neural-network layers' observed activations as a convenient numerical representation of notable visual/phonetic/semantic information in targeted words/expressions, and to use this to group similar leeches together, under the assumption that cards which are highly similar to one another are likely to become leeches because of their mnemonic interference with one another (for example, I have long struggled to remember which of `素直,率直` is read `そっちょく` and which is read `すなお`). Once clustered, the hope is that a student can then more easily look at each cluster separately and use it to derive their own special mnemonics for distinguishing highly similar words/expressions from one another (or in some other way pay these cards special attention).

`example_leeches.txt` is a simple example input file of a few hundred leeches from my own personal study of Japanese. Invoking the manga-ocr-based direct clustering algorithm like:

`python manga_ocr_direct_cluster.py mangaOcrDirectCluster --read_glob_expr="example_leeches.txt" --img_write_folder="leechesDirectTestImgs/" --acts_write_folder="leechesDirectTestActs/"`

Yields:

```
Cluster # 0
        ['かつ', 'だい', 'よー', 'デマ', '便り', '出前', '口実', '口説', '夕日', '姓', '寝台', '岸', '崖', '工夫', '工芸', '幼虫', '当面', '日日', '日陰', '正午', '正
  ', '注', '灯台', '王位', '目安', '直']
Cluster # 1
        ['いっそ', 'うそうそ', 'うろうろ', 'おいで', 'おりゃる', 'ぎっしり', 'ぐいぐい', 'しぶとい', 'すっと', 'ずらり', 'そっくり', 'そっと', 'っきゃ', 'どっと', 'に
  ろ', 'にわか', 'ぱちり', 'ぱっと', 'ふらふら', 'ぶるぶる', 'まがい', 'もはや', 'やみくも', 'パシる', '今にも', '何とも', '例える', '傾らか', '冷める', '四角い', '少
  も', '尻拭い', '微笑む', '恋しい', '手入れ', '根付く', '治める', '見込む', '麗しい']
Cluster # 2
        ['一致', '不平', '中間', '会館', '作法', '便箋', '判事', '半ば', '名残', '問答', '地道', '士業', '好評', '妥当', '実現', '客間', '寒帯', '対', '性質', '恩恵', '成就', '手練', '批評', '投書', '未満', '本命', '標本', '汝等', '漁', '特典', '特色', '献立', '率直', '生垣', '生産', '用途', '番地', '発達', '箇所', '精進', '紙屑', '
  度', '継続', '習字', '芝居', '襖', '説', '課程', '赤道', '足袋', '逆様', '速達', '重役', '針路', '鍔', '餌食', '黙']
Cluster # 3
        ['あまつさえ', 'いざこざ', 'おとしめる', 'これしき', 'ごちゃごちゃ', 'さしかかる', 'さぞかし', 'さもなくば', 'さらさら (更々)', 'しきたり', 'しゃきしゃき', 'す
  きり', 'せっかち', 'そそっかしい', 'ちっとも', 'つったつ', 'どんだけ', 'ぶら下げる', 'やっていく', 'ボコボコ', '一挙手一投足', '何かしら', '使いこなす', '前代未聞', '厚かましい', '引き継ぎ', '引き続く', '引っ掛かる', '張り切る', '思い付く', '思い込み', '性懲りもない', '明明後日', '木漏れ日', '気に留める', '潜り込む', '無何有の郷', '申し付ける', '目をかける', '突き当たり', '突っ走る', '繰り上げる', '草生える', '蛍光灯', '見かける', '言いがかり', '言付ける', '誉れ高い', '通り掛かる', '高慢ちき']
Cluster # 4
        ['~(よ) うと(も)〜(よ)うが', 'いつぞや (何時ぞや)', 'さといも (里芋)', 'ちっと (些と)', 'ひとしきり (一頻り)', 'ましてや (況してや)']
Cluster # 5
        ['一度に', '仕業', '休養', '倦怠期', '健気', '創作', '古文書', '嗅覚', '四つ角', '定規', '家屋', '強気', '快晴', '慰霊祭', '拡充', '支出', '昏倒', '末端', '架
  ', '格別', '概論', '欠陥', '気味', '気品', '気配', '生息', '異例', '直談判', '社説', '積極的', '見出し', '退去', '適用', '間際', '集合']
Cluster # 6
        ['交際', '優位', '唸る', '地盤', '基地', '奇妙', '就寝', '急激', '憤慨', '憤然', '持参', '授与', '掏摸', '敏捷', '整備', '暴動', '書物', '書籍', '演劇', '破く', '素質', '緩和', '表紙', '設備', '課税', '論争', '貪欲', '貫禄', '貯蔵', '貸家', '金銭', '鉄橋', '陰湿']
Cluster # 7
        ['ああ', '今に', '代物', '凍える', '剃る', '嘲る', '大木', '射す', '巧妙', '御名', '所行', '抱える', '敬う', '方角', '更ける', '横切る', '流行る', '現存', '田
  え', '税収', '組む', '統一', '行列', '見入る', '請う', '蹴爪', '青少年', '鷲掴み']
```
