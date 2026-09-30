This project is meant to house modules that explore ways of clustering [Anki (or other SRS program) leeches](https://docs.ankiweb.net/leeches.html) (flashcards the user/student has struggled to remember or otherwise perform on/solve correctly), primarily by grouping them according to measures of visual, phonetic, or semantic similarity. Using a combination of [approximate string matching](https://en.wikipedia.org/wiki/Approximate_string_matching) and various small-to-medium-scale neural networks (e.g., [manga-ocr](https://github.com/kha-white/manga-ocr) and [SmolVLM](https://huggingface.co/blog/smolvlm)) and various preexisting and custom clustering algorithms, I'm currently exploring a few different approaches for visual and phonetic similarity. The first and simplest of these is included in this public GitHub; it combines manga-ocr with the [default KMeans clustering algorithm available in sklearn](https://sklearn.org/stable/modules/generated/sklearn.cluster.KMeans.html). Also included is a more experimental clustering approach that tries to find cluster assignments and the number of clusters automatically by solving a suitable optimziation problem with the [Google OR-Tools CPSAT algorithm](https://developers.google.com/optimization/) (though this currently seems to struggle to find an optimal solution to the rather large and perhaps inefficient optimization model). I expect to update this project in the future as I develop and test additional approaches.

The general idea behind most of these approaches is to use suitable neural-network layers' observed activations as a convenient numerical representation of notable visual/phonetic/semantic information in targeted words/expressions, and to use this to group similar leeches together, under the assumption that cards which are highly similar to one another are likely to become leeches because of their mnemonic interference with one another (for example, I have long struggled to remember which of `素直,率直` is read `そっちょく` and which is read `すなお`). Once clustered, the hope is that a student can then more easily look at each cluster separately and derive their own special mnemonics for distinguishing highly similar words/expressions from one another (or in some other way pay these cards special attention).

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

----

For phonetic similarity, a very rough first pass - with a clustering step not yet added, and currently just using a version of Levenshtein distance weighted by the phonetic dissimilarities [kanasim](https://pypi.org/project/kanasim/) calculated - has been added, which is called (though some of the data inputs are not yet in this repo to enable public runs of it) like:

```
python phonetic_similarity_cluster.py kanasimWeightedLevenshteinCluster --leeches_glob_read_expr="leeches_6_30_2026.txt" --phonetic_dissim_read_path="data/kana_phonetic_dissimilarity_dict.json" --write_path="kanasimPhoneticLevenDists.json" --verbose=False
```

This generates a json file named `kanasimPhoneticLevenDists.json`. Post-processing that slightly on the examples gives output like:

```
掏摸 top 5 -> [('掏摸', 0.0), ('だい', 1.5028306245803833), ('鍔', 1.506257176399231), ('ああ', 1.5566766262054443), ('かつ', 1.5577874183654785)]                     剃る top 5 -> [('剃る', 0.0), ('冷める', 1.7125639915466309), ('射す', 1.7125639915466309), ('ぶら下げる', 1.7125639915466309), ('請う', 1.7281115055084229)]          
冷める top 5 -> [('冷める', 0.0), ('射す', 1.640676736831665), ('ぶら下げる', 1.640676736831665), ('ああ', 1.6948554515838623), ('剃る', 1.7125639915466309)]          
ああ top 5 -> [('ああ', 0.0), ('だい', 1.4230806827545166), ('かつ', 1.445237398147583), ('鍔', 1.4598031044006348), ('崖', 1.5059314966201782)]                       
射す top 5 -> [('射す', 0.0), ('冷める', 1.640676736831665), ('ぶら下げる', 1.640676736831665), ('ああ', 1.6948554515838623), ('剃る', 1.7125639915466309)]            
請う top 5 -> [('請う', 0.0), ('剃る', 1.7281115055084229), ('組む', 1.7370901107788086), ('ああ', 1.7464728355407715), ('更ける', 1.7476577758789062)]                
更ける top 5 -> [('更ける', 0.0), ('組む', 1.6962668895721436), ('かつ', 1.7179936170578003), ('だい', 1.7293221950531006), ('ああ', 1.7401537895202637)]              
組む top 5 -> [('組む', 0.0), ('更ける', 1.6962668895721436), ('かつ', 1.7232029438018799), ('請う', 1.7370901107788086), ('だい', 1.745269536972046)]                 
ぶら下げる top 5 -> [('ぶら下げる', 0.0), ('冷める', 1.640676736831665), ('射す', 1.640676736831665), ('ああ', 1.6948554515838623), ('剃る', 1.7125639915466309)]      
かつ top 5 -> [('かつ', 0.0), ('ああ', 1.445237398147583), ('崖', 1.4599486589431763), ('だい', 1.464479923248291), ('掏摸', 1.5577874183654785)]                      
崖 top 5 -> [('崖', 0.0), ('だい', 1.444204568862915), ('かつ', 1.4599486589431763), ('ああ', 1.5059314966201782), ('掏摸', 1.5805208683013916)]                       
御名 top 5 -> [('御名', 0.0), ('デマ', 1.4573094844818115), ('鍔', 1.515486478805542), ('ああ', 1.5452193021774292), ('掏摸', 1.6469945907592773)]                     
優位 top 5 -> [('優位', 0.0), ('王位', 1.9744336605072021), ('異例', 2.1150872707366943), ('餌食', 2.188729763031006), ('出前', 2.1929454803466797)]                   
デマ top 5 -> [('デマ', 0.0), ('鍔', 1.4489266872406006), ('御名', 1.4573094844818115), ('ああ', 1.5200345516204834), ('だい', 1.6125433444976807)]                    
だい top 5 -> [('だい', 0.0), ('ああ', 1.4230806827545166), ('崖', 1.444204568862915), ('かつ', 1.464479923248291), ('掏摸', 1.5028306245803833)]                      
よー top 5 -> [('よー', 0.0), ('だい', 1.7744851112365723), ('ああ', 1.7757227420806885), ('崖', 1.7777578830718994), ('かつ', 1.7842270135879517)]                    
鍔 top 5 -> [('鍔', 0.0), ('デマ', 1.4489266872406006), ('ああ', 1.4598031044006348), ('掏摸', 1.506257176399231), ('御名', 1.515486478805542)]                        
王位 top 5 -> [('王位', 0.0), ('優位', 1.9744336605072021), ('おいで', 2.039766550064087), ('異例', 2.04660701751709), ('餌食', 2.113097906112671)]     
```

On a quick inspection, none of these look very similar to one another. It may be that in this particular batch of leeches phonetic similarity is just not a large driver of mnemonic interference, but additional work is needed (e.g., currently this approach only uses weights associated with the kanasim-computed single-character substitutions, but approximate string matching can accommodate multiple-character substitutions as well, and kanasim provides weights for many "biphones").
