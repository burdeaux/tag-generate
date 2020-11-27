def loadtxt(filename):
    txt=[]
    with open(filename,encoding="utf-8") as f:
        for line in f :
            txt.append(line.rstrip())
    return txt

class Vocab(object):
    "map symbols(word/tokens) to indices"
    def __init__(self):
        #containers
        self.symbols =[]#build dictionary from ids to symbols
        self.idxs={}#build dictionary from symbols to ids
        #frozen state 控制不增加新字典,默认false
        self.frozen = False
        #special symbols
        self.add_symbol["<pad>"] #Padding
        self.add_symbol["<sos>"]
        self.add_symbol["<eos>"]
        self.add_symbol["<unk>"]
    def __len__(self):##len(self)调用
        return len(self.idxs)
    def add_symbol(self,symbol):
        """add a symbol to the dictionary and return its index
        if the symbol already exists only return the index """
        if symbol not in self.idxs:
            if self.frozen:
                raise ValueError("Can't add symbol to frozen dictionary")
            self.symbols.append(symbol)
            self.idxs[symbol]=len(self.idxs)#字典中加入新元素的方法，利用列表append和len，也可以使用self.idxs.update({})单这样不方便计算index值
        return self.idxs[symbol]#分支任务if即可，共同任务放if外
    def to_idx(self,symbol):#返回symbol的idx,如果不在字典则unk
        if symbol in self.idxs:
            return self.idxs[symbol]
        else:
            return self.idxs["unk"]
    def to_symbol(self,idx):
        """return id's symbol"""
        return self.symbols[idx] ##symbols列表对应id-symbol,字典idx{}对应symbol-id
    def __getitem__(self,symbol_or_idx):
        if isinstance(symbol_or_idx,int):
            return self.to_symbol(symbol_or_idx)#isinstance(self,classinfo) 判断一个实例是否为特定类型，int返回对应symbol,否则(string)返回idx
        else:
            return self.to_idx(symbol_or_idx)
    @staticmethod
    def from_data_files(*filenames,max_size=-1,min_freq=2):#*和**表任意个数参数，*表示以元组形式如foo(1,2,3,4)，**表示以字典形式foo(1,a=3,b=4)
        """builds a dictionary from the most frequent tokens in files"""
        vocab = Vocab()
        #Record token counts
        token_counts = defaultdict(lambda:0)#高级一点的dict,key不存在时返回默认值 lambda:0相当于函数def p() return 0，用于初始化一个defaultdict()
        for filename in filenames:
            with open(filename,encoding='utf-8') as f:
                for line in f:
                    tokens= line.rstrip().split() #列表记录每个单词
                    for token in tokens:
                        token_counts[token] += 1
        #filter out least frequent tokens
        token_counts = {
            tok:cnt
            for tok,cnt in token_counts.items()
            if cnt >=min_freq    #字典或列表的简单遍历过滤，使用.items()返回可遍历的元组数据(key,value)
        }
        #only keep most common tokens 
        tokens = list(token_counts.keys())#list() 元组换成list，元组数值没法改，实际上keys()已经返回列表了
        sorted_tokens = sorted(tokens,key=lambda x:token_counts[x])[::-1]#sorted创建新列，sort改原list,sort仅用于列表，推荐使用sorted,key表示用于排序的值，[::-1]简写倒排
        #add the remaining tokens to the dictionary
        for token in sorted_tokens:
            vocab.add_symbol(token)
        return vocab
    
###以上为Vocab,建立一个idx-symbol的map
def _make_tagged_tokens(sents,pad_idx):
    """pad sentences to the max length and create the relevant tag"""
    lengths = [len(sent) for sent in sents]#每个sent的长度list,用于padding切片
    max_len = max(lengths)#max可接受元组或者list
    bsz = len(lengths) #list长度即是batch size
    #Tensor containing the (right) padded tokens
    tokens = th.full((max_len,bsz,pad_idx)).long()#full(size,value),pad先全填，再改值容易实现
    for i in range(bsz):
        tokens[:lengths[i]，i]=th.LongTensor(sents[i])
    #mask such that tag[i,b] = 1 if i > lengths[b]，即用1标注pad部分  
    lengths = th.LongTensor(lengths).view(1,-1) #1*bsz
    tag=th.gt(th.arange(max_len).view(-1,1),lengths)#th.arrange()返回一个[0,max_len)不含max_len一共max_len位的tensor,th.gt greater than #tag需要的sz为i*b,用th.gt进行pad,每个batch行对应length[b],即pad部分为1，实体字段为0
    return tokens, tag

class MTDataset(data.Dataset):
    def __init__(self,vocab,prefix,src_lang="en",tgt_lang="fr"):
        #Attributes
        self.vocab = vocab
        self.src_lang = src_lang #平行模型中的source标志
        self.tgt_lang = tgt_lang #平行模型中的target标志
        #Load from files, 文件名entagged/engenerated_parallel.bpe.dev.en/tagged
        src_file = prefix + "."+ src_lang #字符串拼接
        tgt_file = prefix + "."+ tgt_lang
        #check length(source和tgt是否对应)
        self.length = len(self.src_txt)
        if self.length != len(self.tgt_txt):
            raise ValueError("Mismatched source and target length")
        #Append start/end of snetencen token to the target
        for 