"""针对call2018数据集提供的数据处理工具"""
import json
import time
from typing import Tuple, List, Dict

import torch

from setting import TRAIN_JSON_PATH, TEST_JSON_PATH, REST_DATA_TXT_PATH, LAW_TXT_PATH, ACCUSATION_TXT_PATH, \
    TRAIN_JSON_LEN, TEST_JSON_LEN, REST_DATA__JSON_LEN
from torch.utils.data import Dataset, DataLoader

# 因为Fact和罪名/法条不是一一对应的 所以需要进一步的处理
SIMPLE_DEBUG = True


class JsonReader:
    """针对call2018的json文件读取对象"""

    def __init__(self, JSON_PATH: str, LEN: int, encoding: str = 'utf-8'):
        self.json_path = JSON_PATH
        self.encoding = encoding
        self.len = LEN
        self.data: List[Dict] = []
        with open(self.json_path, encoding=self.encoding) as file:
            for index, line in enumerate(file, 0):
                line_dict = json.loads(line)
                meta = line_dict['meta']
                if len(meta['relevant_articles']) != 1 or len(meta['accusation']) != 1:
                    # 代表有多个相关法条 或者 多个指控
                    # 目前采用最简单的方法 遇到这种案件就略过
                    continue
                else:
                    self.data.append(line_dict)

    def __getitem__(self, item):
        # 这一步效率很低 但是没有好的办法
        return self.data[item]

    def __len__(self):
        return self.len


class Call2018Dataset(Dataset):
    """
    Call2018法律数据集
    当前能做的事情: 获取fact字段和label字段  fact暂时没办法转成 Tensor
    当前数据集的目标 - 获取fact 和 相关法条 pair
    """

    def __init__(self, json_path: str, json_len: int, law_path, accusation_path, encoding: str = 'utf-8',
                 field: str = 'accusation'):
        # 获取文件行数量
        self.json_reader = JsonReader(json_path, json_len, encoding)
        self.field = field
        self.law2index, self.acc2index, self.index2law, self.index2acc = self._get_law_dict(law_path, accusation_path)

    def __getitem__(self, item):
        fact, target = self._getitem_(item)
        # _ToDo 对fact进行分词 然后将 词列表 转换成 词索引列表

        if self.field == 'accusation':
            # (对accusation List进行Tensor化) 简化处理下 对accusation List的元素只有1个
            pass
        if self.field == 'relevant_articles':
            # (对 相关法条 List进行 Tensor化) 简化处理下 相关法条 List的元素只有1个
            target = torch.tensor((int(target[0])), dtype=torch.long)
        return fact, target

    def _getitem_(self, item):
        if self.field == 'accusation':  # 事实-罪名数据
            return self.json_reader[item]['fact'], self.json_reader[item]['meta'][self.field]
        elif self.field == 'imprisonment':  # 事实-刑期数据
            return self.json_reader[item]['fact'], self.json_reader[item]['meta']['term_of_imprisonment'][self.field]
        elif self.field == 'death_penalty':  # 事实-死刑数据
            return self.json_reader[item]['fact'], self.json_reader[item]['meta']['term_of_imprisonment'][self.field]
        elif self.field == 'life_imprisonment':  # 事实-终生监禁数据
            return self.json_reader[item]['fact'], self.json_reader[item]['meta']['term_of_imprisonment'][self.field]
        elif self.field == 'relevant_articles':  # 事实-相关法条数据
            return self.json_reader[item]['fact'], self.json_reader[item]['meta'][self.field]
        elif self.field == 'criminals':  # 事实 -罪犯数据
            return self.json_reader[item]['fact'], self.json_reader[item]['meta'][self.field]
        elif self.field == 'punish_of_money':  # 事实-罚款数据
            return self.json_reader[item]['fact'], self.json_reader[item]['meta'][self.field]

    @classmethod
    def _get_law_dict(cls, law_path, acc_path):
        law2index, acc2index = {}, {}
        index2law, index2acc = {}, {}
        with open(law_path, encoding='utf-8') as law_file:
            for index, law in enumerate(law_file, 0):
                law = int(law)
                law2index[law], index2law[index] = index, law
        with open(acc_path, encoding='utf-8') as acc_file:
            for index, acc in enumerate(acc_file, 0):
                acc = acc.strip()
                acc2index[acc], index2acc[index] = index, acc

        return law2index, acc2index, index2law, index2acc

    def __len__(self):
        return len(self.json_reader)


if __name__ == '__main__':
    dataset = Call2018Dataset(TEST_JSON_PATH, TEST_JSON_LEN,
                              LAW_TXT_PATH, ACCUSATION_TXT_PATH, field='relevant_articles')
    dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=1)
    for fact, relevant_article in dataset:
        print(relevant_article)
        break
