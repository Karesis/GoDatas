#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gogames数据集专用整理工具
专门用于整理https://homepages.cwi.nl/~aeb/go/games/的多层嵌套SGF文件集合
"""

import os
import sys
import re
import shutil
import hashlib
import argparse
import logging
import json
import unicodedata
import concurrent.futures
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gogames_organizer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("gogames_organizer")

# 棋手信息数据库
PLAYER_DB = {
    # 中国
    'ke jie': {'name': 'Ke Jie', 'rank': '9p', 'country': 'China', 'aliases': ['柯洁', 'ke jie']},
    'gu li': {'name': 'Gu Li', 'rank': '9p', 'country': 'China', 'aliases': ['古力', 'gu li']},
    'chang hao': {'name': 'Chang Hao', 'rank': '9p', 'country': 'China', 'aliases': ['常昊', 'chang hao']},
    'nie weiping': {'name': 'Nie Weiping', 'rank': '9p', 'country': 'China', 'aliases': ['聂卫平', 'nie weiping']},
    'ma xiaochun': {'name': 'Ma Xiaochun', 'rank': '9p', 'country': 'China', 'aliases': ['马晓春', 'ma xiaochun']},
    'mi yuting': {'name': 'Mi Yuting', 'rank': '9p', 'country': 'China', 'aliases': ['芈昱廷', 'mi yuting']},
    'fan tingyu': {'name': 'Fan Tingyu', 'rank': '9p', 'country': 'China', 'aliases': ['范廷钰', 'fan tingyu']},
    'tang weixing': {'name': 'Tang Weixing', 'rank': '9p', 'country': 'China', 'aliases': ['唐韦星', 'tang weixing']},
    'yang dingxin': {'name': 'Yang Dingxin', 'rank': '9p', 'country': 'China', 'aliases': ['杨鼎新', 'yang dingxin']},
    'shi yue': {'name': 'Shi Yue', 'rank': '9p', 'country': 'China', 'aliases': ['时越', 'shi yue']},
    'tan xiao': {'name': 'Tan Xiao', 'rank': '9p', 'country': 'China', 'aliases': ['檀啸', 'tan xiao']},
    'lian xiao': {'name': 'Lian Xiao', 'rank': '9p', 'country': 'China', 'aliases': ['连笑', 'lian xiao']},
    'zhou ruiyang': {'name': 'Zhou Ruiyang', 'rank': '9p', 'country': 'China', 'aliases': ['周睿羊', 'zhou ruiyang']},
    'peng liyao': {'name': 'Peng Liyao', 'rank': '7p', 'country': 'China', 'aliases': ['彭立尧', 'peng liyao']},
    
    # 韩国
    'lee sedol': {'name': 'Lee Sedol', 'rank': '9p', 'country': 'Korea', 'aliases': ['李世石', 'yi sedol', 'lee sedol', 'ri sedol', 'lee se-dol']},
    'park junghwan': {'name': 'Park Junghwan', 'rank': '9p', 'country': 'Korea', 'aliases': ['朴廷桓', 'park junghwan', 'park jeong-hwan']},
    'shin jinseo': {'name': 'Shin Jinseo', 'rank': '9p', 'country': 'Korea', 'aliases': ['申真谞', 'shin jinseo', 'shin jin seo']},
    'kang dongyun': {'name': 'Kang Dongyun', 'rank': '9p', 'country': 'Korea', 'aliases': ['姜东润', 'kang dongyun']},
    'park younghun': {'name': 'Park Younghun', 'rank': '9p', 'country': 'Korea', 'aliases': ['朴永训', 'park younghun']},
    'choi cheolhan': {'name': 'Choi Cheolhan', 'rank': '9p', 'country': 'Korea', 'aliases': ['崔哲瀚', 'choi cheolhan']},
    'cho hanseung': {'name': 'Cho Hanseung', 'rank': '9p', 'country': 'Korea', 'aliases': ['赵汉乘', 'cho hanseung']},
    'won sungjin': {'name': 'Won Sungjin', 'rank': '9p', 'country': 'Korea', 'aliases': ['元晟溱', 'won sungjin']},
    'lee changho': {'name': 'Lee Changho', 'rank': '9p', 'country': 'Korea', 'aliases': ['李昌镐', 'lee changho', 'yi changho', 'lee chang-ho']},
    'kim jiseok': {'name': 'Kim Jiseok', 'rank': '9p', 'country': 'Korea', 'aliases': ['金志锡', 'kim jiseok']},
    'byun sangil': {'name': 'Byun Sangil', 'rank': '9p', 'country': 'Korea', 'aliases': ['卞相壹', 'byun sangil']},
    'lee donghoon': {'name': 'Lee Donghoon', 'rank': '9p', 'country': 'Korea', 'aliases': ['李东勋', 'lee donghoon']},
    
    # 日本
    'iyama yuta': {'name': 'Iyama Yuta', 'rank': '9p', 'country': 'Japan', 'aliases': ['井山裕太', 'iyama yuta']},
    'cho u': {'name': 'Cho U', 'rank': '9p', 'country': 'Japan', 'aliases': ['张栩', 'cho u']},
    'yamashita keigo': {'name': 'Yamashita Keigo', 'rank': '9p', 'country': 'Japan', 'aliases': ['山下敬吾', 'yamashita keigo']},
    'ichiriki ryo': {'name': 'Ichiriki Ryo', 'rank': '8p', 'country': 'Japan', 'aliases': ['一力遼', 'ichiriki ryo']},
    'yu zhengqi': {'name': 'Yu Zhengqi', 'rank': '8p', 'country': 'Japan', 'aliases': ['余正麒', 'yu zhengqi']},
    'xie jiayuan': {'name': 'Xie Jiayuan', 'rank': '8p', 'country': 'Japan', 'aliases': ['許家元', 'xie jiayuan']},
    'shibano toramaru': {'name': 'Shibano Toramaru', 'rank': '9p', 'country': 'Japan', 'aliases': ['芝野虎丸', 'shibano toramaru']},
    'takao shinji': {'name': 'Takao Shinji', 'rank': '9p', 'country': 'Japan', 'aliases': ['高尾紳路', 'takao shinji']},
    'cho chikun': {'name': 'Cho Chikun', 'rank': '9p', 'country': 'Japan', 'aliases': ['趙治勲', 'cho chikun']},
    'takemiya masaki': {'name': 'Takemiya Masaki', 'rank': '9p', 'country': 'Japan', 'aliases': ['武宮正樹', 'takemiya masaki']},
    'kobayashi koichi': {'name': 'Kobayashi Koichi', 'rank': '9p', 'country': 'Japan', 'aliases': ['小林光一', 'kobayashi koichi']}
}

# 著名比赛/赛事信息 - 扩展以匹配gogames数据集
TOURNAMENT_DB = {
    # 国际比赛
    'samsung cup': {'name': 'Samsung Cup', 'full_name': 'Samsung Fire & Marine Insurance Cup World Baduk Masters'},
    'lg cup': {'name': 'LG Cup', 'full_name': 'LG Cup International'},
    'chunlan cup': {'name': 'Chunlan Cup', 'full_name': 'Chunlan Cup World Professional Go Championship'},
    'fujitsu cup': {'name': 'Fujitsu Cup', 'full_name': 'Fujitsu Cup World Go Championship'},
    'ing cup': {'name': 'Ing Cup', 'full_name': 'Ing Cup Professional Tournament'},
    'bc card cup': {'name': 'BC Card Cup', 'full_name': 'BC Card Cup World Baduk Championship'},
    'mlily cup': {'name': 'MLily Cup', 'full_name': 'MLily Cup World Go Open Tournament'},
    'bailing cup': {'name': 'Bailing Cup', 'full_name': 'Bailing Cup World Go Championship'},
    'lotte cup': {'name': 'Lotte Cup', 'full_name': 'Lotte Cup World Women\'s Baduk Championship'},
    'nongshim cup': {'name': 'Nongshim Cup', 'full_name': 'Nongshim Cup World Professional Weiqi Championship'},
    'hokuto cup': {'name': 'Hokuto Cup', 'full_name': 'Hokuto Cup International Go Tournament'},
    
    # 日本比赛
    'meijin': {'name': 'Meijin', 'full_name': 'Meijin Tournament'},
    'honinbo': {'name': 'Honinbo', 'full_name': 'Honinbo Tournament'},
    'kisei': {'name': 'Kisei', 'full_name': 'Kisei Tournament'},
    'tengen': {'name': 'Tengen', 'full_name': 'Tengen Tournament'},
    'oza': {'name': 'Oza', 'full_name': 'Oza Tournament'},
    'judan': {'name': 'Judan', 'full_name': 'Judan Tournament'},
    'gosei': {'name': 'Gosei', 'full_name': 'Gosei Tournament'},
    'ryusei': {'name': 'Ryusei', 'full_name': 'Ryusei Tournament'},
    'agon cup': {'name': 'Agon Cup', 'full_name': 'Agon Kiriyama Cup'},

    # 中国比赛
    'mingren': {'name': 'Mingren', 'full_name': 'Mingren Tournament'},
    'tianyuan': {'name': 'Tianyuan', 'full_name': 'Tianyuan Tournament'},
    'cctv cup': {'name': 'CCTV Cup', 'full_name': 'CCTV Cup'},
    'xingqi': {'name': 'CCTV Cup', 'full_name': 'CCTV Cup'},
    'qisheng': {'name': 'Qisheng', 'full_name': 'Qisheng Tournament'},
    'guoshou': {'name': 'Guoshou', 'full_name': 'Guoshou Tournament'},
    'changqi cup': {'name': 'Changqi Cup', 'full_name': 'Changqi Cup'},
    
    # 韩国比赛
    'kuksu': {'name': 'Kuksu', 'full_name': 'Kuksu Tournament'},
    'myungin': {'name': 'Myungin', 'full_name': 'Myungin Tournament'},
    'kbs cup': {'name': 'KBS Cup', 'full_name': 'KBS Cup'},
    'maxim cup': {'name': 'Maxim Cup', 'full_name': 'Maxim Cup'},
    'sajik cup': {'name': 'Sajik Cup', 'full_name': 'Sajik Cup'},
    'bc card cup': {'name': 'BC Card Cup', 'full_name': 'BC Card Cup'},
    'paewang': {'name': 'Paewang', 'full_name': 'Paewang Tournament'},
    
    # 特殊标记的赛事 - 从gogames分析
    'daiwa': {'name': 'Daiwa Cup', 'full_name': 'Daiwa Cup'},
    'daiwa gc': {'name': 'Daiwa Grand Champion Cup', 'full_name': 'Daiwa Grand Champion Cup'},
    'fujitsu': {'name': 'Fujitsu Cup', 'full_name': 'Fujitsu Cup'},
    'samsung': {'name': 'Samsung Cup', 'full_name': 'Samsung Cup'},
    'kiwang': {'name': 'Kiwang', 'full_name': 'Kiwang Tournament'},
    'female': {'name': 'Female', 'full_name': 'Female Tournament'},
    'world': {'name': 'World Championship', 'full_name': 'World Go Championship'},
    'international': {'name': 'International Tournament', 'full_name': 'International Go Tournament'},
    'pro best ten': {'name': 'Pro Best Ten', 'full_name': 'Professional Best Ten Tournament'},
    'leeku': {'name': 'Lee-Ku Match', 'full_name': 'Lee Changho vs Ku Li Match'},
    'tvasia': {'name': 'TV Asia Cup', 'full_name': 'TV Asia Championship'},
    'tong yang': {'name': 'Tong Yang Cup', 'full_name': 'Tong Yang Securities Cup'},
    'bacchus': {'name': 'Bacchus Cup', 'full_name': 'Bacchus Cup'}
}

# 目录映射到赛事
DIR_TO_TOURNAMENT = {
    'daiwagc': 'Daiwa Grand Champion Cup',
    'daiwa': 'Daiwa Cup',
    'fujitsu': 'Fujitsu Cup',
    'honinbo': 'Honinbo',
    'kisei': 'Kisei',
    'meijin': 'Meijin',
    'judan': 'Judan',
    'samsung': 'Samsung Cup',
    'lg': 'LG Cup',
    'chunlan': 'Chunlan Cup',
    'ing': 'Ing Cup',
    'nongshim': 'Nongshim Cup',
    'female': 'Female Tournament',
    'misc': 'Miscellaneous Games',
    'kiwang': 'Kiwang',
    'leeku': 'Lee-Ku Match',
    'ryusei': 'Ryusei',
    'myungin': 'Myungin',
    'kuksu': 'Kuksu',
    'oza': 'Oza',
    'gosei': 'Gosei',
    'bc': 'BC Card Cup',
    'tvasia': 'TV Asia Cup',
    'pro': 'Professional Games'
}


class GogamesOrganizer:
    """Gogames数据集整理器"""
    
    def __init__(self, input_dir, output_dir, min_quality=70, organize_by='quality', 
                 fix_metadata=True, remove_duplicates=True, extract_from_path=True,
                 workers=4, verbose=False):
        """
        初始化整理器
        
        参数:
            input_dir (str): 输入目录，包含杂乱SGF文件
            output_dir (str): 输出目录，用于保存整理后的文件
            min_quality (int): 最低质量阈值(0-100)
            organize_by (str): 组织方式，可选 'quality', 'player', 'date', 'event'
            fix_metadata (bool): 是否尝试修复和标准化元数据
            remove_duplicates (bool): 是否移除重复文件
            extract_from_path (bool): 是否从文件路径提取额外元数据
            workers (int): 并行处理的工作线程数
            verbose (bool): 是否显示详细日志
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.min_quality = min_quality
        self.organize_by = organize_by
        self.fix_metadata = fix_metadata
        self.remove_duplicates = remove_duplicates
        self.extract_from_path = extract_from_path
        self.workers = workers
        self.verbose = verbose
        
        # 设置日志级别
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # 统计信息
        self.stats = {
            'total_files': 0,
            'valid_sgf_files': 0,
            'invalid_files': 0,
            'duplicates': 0,
            'fixed_metadata': 0,
            'high_quality': 0,
            'low_quality': 0,
            'pro_games': 0,
            'amateur_games': 0,
            'unknown_type': 0,
            'games_by_player': defaultdict(int),
            'games_by_event': defaultdict(int),
            'games_by_year': defaultdict(int),
            'games_by_quality': defaultdict(int),
            'metadata_from_path': 0,
            'directory_distribution': defaultdict(int)
        }
        
        # 用于重复检测的哈希集合
        self.file_hashes = set()
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def find_sgf_files(self):
        """查找所有SGF文件"""
        sgf_files = []
        
        # 递归查找所有.sgf文件
        for file_path in self.input_dir.rglob('*.sgf'):
            sgf_files.append(file_path)
            
            # 统计目录分布
            parent_dir = file_path.parent.name.lower()
            grandparent_dir = file_path.parent.parent.name.lower()
            self.stats['directory_distribution'][grandparent_dir] += 1
            
        logger.info(f"在 {self.input_dir} 中找到 {len(sgf_files)} 个SGF文件")
        self.stats['total_files'] = len(sgf_files)
        return sgf_files
    
    def is_duplicate(self, file_path):
        """检查文件是否是重复的"""
        if not self.remove_duplicates:
            return False
            
        # 计算文件哈希值
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
            
        # 检查是否已存在此哈希值
        if file_hash in self.file_hashes:
            self.stats['duplicates'] += 1
            return True
            
        # 添加到哈希集合
        self.file_hashes.add(file_hash)
        return False
    
    def extract_metadata_from_path(self, file_path):
        """从文件路径提取元数据"""
        if not self.extract_from_path:
            return {}
            
        metadata = {}
        
        # 从文件名提取信息
        filename = file_path.stem
        
        # 一些常见的文件名格式:
        # 1. player1-player2
        # 2. player1_player2
        # 3. player1_player2_event
        # 4. player1_player2_date
        # 5. player1-vs-player2
        
        # 尝试提取棋手信息
        name_parts = re.split(r'[-_\s]', filename)
        if len(name_parts) >= 2:
            metadata['black_player'] = name_parts[0]
            metadata['white_player'] = name_parts[1]
        
        # 从父目录和祖父目录提取赛事信息
        parent_dir = file_path.parent.name.lower()
        grandparent_dir = file_path.parent.parent.name.lower()
        
        # 尝试通过目录名匹配赛事
        for dir_name in [grandparent_dir, parent_dir]:
            # 检查目录名是否映射到已知赛事
            for key, tournament_name in DIR_TO_TOURNAMENT.items():
                if key in dir_name.lower():
                    metadata['event'] = tournament_name
                    break
                    
            # 如果已找到赛事，跳出循环
            if 'event' in metadata:
                break
        
        # 尝试从文件名或目录名提取日期
        date_match = re.search(r'(\d{4})[-\.]?(\d{1,2})?[-\.]?(\d{1,2})?', filename)
        if not date_match:
            # 尝试在目录名中查找
            date_match = re.search(r'(\d{4})[-\.]?(\d{1,2})?[-\.]?(\d{1,2})?', parent_dir)
        
        if date_match:
            year = date_match.group(1)
            month = date_match.group(2) if date_match.group(2) else '01'
            day = date_match.group(3) if date_match.group(3) else '01'
            metadata['date'] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        
        # 如果成功提取了任何元数据
        if metadata:
            self.stats['metadata_from_path'] += 1
            
        return metadata
    
    def parse_sgf(self, file_path):
        """解析SGF文件内容，提取元数据"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # 检查文件是否是有效的SGF格式
            if not content.startswith('(;') or (';B[' not in content and ';W[' not in content):
                self.stats['invalid_files'] += 1
                return None, content
                
            # 提取元数据
            metadata = {}
            patterns = {
                'event': r'EV\[([^\]]*)\]',
                'date': r'DT\[([^\]]*)\]',
                'black_player': r'PB\[([^\]]*)\]',
                'white_player': r'PW\[([^\]]*)\]',
                'result': r'RE\[([^\]]*)\]',
                'black_rank': r'BR\[([^\]]*)\]',
                'white_rank': r'WR\[([^\]]*)\]',
                'komi': r'KM\[([^\]]*)\]',
                'rules': r'RU\[([^\]]*)\]',
                'time_limit': r'TM\[([^\]]*)\]',
                'place': r'PC\[([^\]]*)\]',
                'comment': r'C\[([^\]]*)\]',
                'copyright': r'CP\[([^\]]*)\]',
                'handicap': r'HA\[([^\]]*)\]',
                'game_name': r'GN\[([^\]]*)\]',
                'application': r'AP\[([^\]]*)\]',
                'game_id': r'GC\[([^\]]*)\]',
                'source': r'SO\[([^\]]*)\]'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, content)
                if match:
                    metadata[key] = match.group(1).strip()
            
            # 如果缺少基本信息，尝试从路径提取
            if 'black_player' not in metadata or 'white_player' not in metadata or 'event' not in metadata:
                path_metadata = self.extract_metadata_from_path(file_path)
                
                # 只添加缺失的字段
                for key, value in path_metadata.items():
                    if key not in metadata or not metadata[key]:
                        metadata[key] = value
            
            self.stats['valid_sgf_files'] += 1
            return metadata, content
        except Exception as e:
            logger.error(f"解析文件时出错 {file_path}: {e}")
            self.stats['invalid_files'] += 1
            return None, None
    
    def fix_player_names(self, metadata):
        """修复和标准化棋手名称"""
        if not self.fix_metadata:
            return metadata
            
        for player_key in ['black_player', 'white_player']:
            if player_key not in metadata:
                continue
                
            player_name = metadata[player_key]
            
            # 移除非ASCII字符
            ascii_name = ''.join(c for c in unicodedata.normalize('NFD', player_name) 
                                if unicodedata.category(c) != 'Mn')
            
            # 尝试在玩家数据库中查找
            lookup_name = ascii_name.lower().strip()
            
            # 检查是否在数据库中
            for key, player_data in PLAYER_DB.items():
                if lookup_name in player_data['aliases'] or key in lookup_name:
                    # 使用标准化的名称
                    metadata[player_key] = player_data['name']
                    # 添加或修正段位信息
                    rank_key = f"{'black' if player_key == 'black_player' else 'white'}_rank"
                    if rank_key not in metadata or not metadata[rank_key]:
                        metadata[rank_key] = player_data['rank']
                    # 添加国家信息
                    country_key = f"{'black' if player_key == 'black_player' else 'white'}_country"
                    metadata[country_key] = player_data['country']
                    
                    # 将该棋手的比赛计数+1
                    self.stats['games_by_player'][player_data['name']] += 1
                    break
        
        return metadata
    
    def fix_date(self, metadata):
        """修复和标准化日期格式"""
        if not self.fix_metadata or 'date' not in metadata:
            return metadata
            
        date_str = metadata['date']
        
        # 尝试多种日期格式
        date_patterns = [
            # YYYY-MM-DD
            r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})',
            # DD.MM.YYYY
            r'(\d{1,2})\.(\d{1,2})\.(\d{4})',
            # YYYY.MM.DD
            r'(\d{4})\.(\d{1,2})\.(\d{1,2})',
            # 只有年份
            r'(\d{4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, date_str)
            if match:
                if len(match.groups()) == 3:  # 完整日期
                    if '.' in pattern:  # 欧洲格式日期
                        day, month, year = match.groups()
                    else:  # ISO格式
                        year, month, day = match.groups()
                    # 确保月和日是两位数
                    month = month.zfill(2)
                    day = day.zfill(2)
                    metadata['date'] = f"{year}-{month}-{day}"
                else:  # 只有年份
                    year = match.group(1)
                    metadata['date'] = f"{year}-01-01"  # 默认为该年的1月1日
                
                # 记录年份统计
                self.stats['games_by_year'][year] += 1
                break
        
        return metadata
    
    def fix_event(self, metadata):
        """修复和标准化赛事信息"""
        if not self.fix_metadata or 'event' not in metadata:
            return metadata
            
        event_str = metadata['event'].lower()
        
        # 查找已知的锦标赛
        for key, tournament in TOURNAMENT_DB.items():
            if key in event_str:
                metadata['event'] = tournament['name']
                metadata['event_full'] = tournament['full_name']
                self.stats['games_by_event'][tournament['name']] += 1
                return metadata
        
        # 如果没有匹配到已知赛事，尝试规范化现有的赛事名称
        if metadata['event']:
            # 保留到最长50个字符，避免过长的赛事名称
            metadata['event'] = metadata['event'][:50]
            self.stats['games_by_event'][metadata['event']] += 1
        
        return metadata
    
    def evaluate_game_quality(self, metadata, file_path):
        """评估对局质量"""
        score = 50  # 基础分
        
        # 职业对局加分
        is_pro = False
        for rank_key in ['black_rank', 'white_rank']:
            if rank_key in metadata:
                rank = metadata[rank_key]
                if rank and ('p' in rank.lower() or rank.endswith('段') or 'd' in rank.lower()):
                    # 段位评估
                    try:
                        # 提取数字部分
                        rank_num = int(re.search(r'\d+', rank).group())
                        if 'p' in rank.lower():  # 职业棋手
                            score += min(rank_num * 3, 20)  # 最多加20分
                            is_pro = True
                        else:  # 业余棋手
                            score += min(rank_num, 10)  # 最多加10分
                    except (AttributeError, ValueError):
                        # 无法提取数字，但仍视为段位
                        score += 5
        
        # 职业对局总加分
        if is_pro:
            score += 10
            self.stats['pro_games'] += 1
        else:
            self.stats['amateur_games'] += 1
        
        # 判断棋手知名度
        for player_key in ['black_player', 'white_player']:
            if player_key in metadata:
                player_name = metadata[player_key].lower()
                for key, player_data in PLAYER_DB.items():
                    if player_name in player_data['aliases'] or key in player_name:
                        score += 15
                        # 如果是著名棋手，可能是职业对局
                        if not is_pro:
                            score += 5
                            is_pro = True
                        break
        
        # 赛事加分
        if 'event' in metadata and metadata['event']:
            event = metadata['event'].lower()
            # 国际大赛或重要锦标赛
            if any(key in event for key in TOURNAMENT_DB.keys()):
                score += 15
            # 其他锦标赛
            elif any(term in event for term in ['cup', 'title', 'tournament', 'championship', '赛', '杯', '冠军']):
                score += 10
        
        # 文件路径加分 - 从特定目录加分
        path_str = str(file_path).lower()
        if any(key in path_str for key in DIR_TO_TOURNAMENT.keys()):
            score += 10
        
        # 根据日期加分
        if 'date' in metadata and metadata['date']:
            try:
                year = int(metadata['date'].split('-')[0])
                current_year = datetime.now().year
                if year >= (current_year - 5):
                    score += 5  # 最近5年的棋谱加分
            except (IndexError, ValueError):
                pass
        
        # 文件是否包含注释/变化图
        if 'comment' in metadata and metadata['comment']:
            score += 10  # 有注释的棋谱更有教学价值
        
        # 应用质量阈值
        quality = min(100, score)
        metadata['quality'] = quality
        
        # 记录质量统计
        quality_bracket = (quality // 10) * 10  # 分为0-9, 10-19, ..., 90-100等档位
        self.stats['games_by_quality'][quality_bracket] += 1
        
        if quality >= self.min_quality:
            self.stats['high_quality'] += 1
        else:
            self.stats['low_quality'] += 1
            
        return metadata, quality
    
    def create_filename(self, metadata):
        """根据元数据创建规范化的文件名"""
        # 提取日期
        date = metadata.get('date', '0000-00-00')
        if not date or date == '0000-00-00':
            date = datetime.now().strftime('%Y-%m-%d')
        
        # 提取棋手
        black = re.sub(r'[\\/*?:"<>|]', '', metadata.get('black_player', 'unknown')).strip()
        white = re.sub(r'[\\/*?:"<>|]', '', metadata.get('white_player', 'unknown')).strip()
        
        # 提取赛事
        event = re.sub(r'[\\/*?:"<>|]', '', metadata.get('event', '')).strip()
        if not event:
            event = 'Game'
        
        # 格式化结果
        result = metadata.get('result', '')
        if result:
            result = re.sub(r'[\\/*?:"<>|]', '', result).strip()
            result = f"_Result-{result}"
        else:
            result = ""
        
        # 限制长度
        black = black[:20]
        white = white[:20]
        event = event[:30]
        
        # 构建文件名
        quality = metadata.get('quality', 0)
        filename = f"{date}_{black}_vs_{white}_{event}{result}_Q{quality}.sgf"
        # 替换Windows文件名非法字符
        filename = re.sub(r'[\\/*?:"<>|]', '_', filename)
        return filename
    
    def process_sgf_file(self, file_path):
        """处理单个SGF文件"""
        try:
            # 检查是否是重复文件
            if self.is_duplicate(file_path):
                logger.debug(f"跳过重复文件: {file_path}")
                return None
            
            # 解析SGF文件
            metadata, content = self.parse_sgf(file_path)
            if metadata is None:
                logger.debug(f"无效的SGF文件: {file_path}")
                return None
            
            # 修复元数据
            if self.fix_metadata:
                metadata = self.fix_player_names(metadata)
                metadata = self.fix_date(metadata)
                metadata = self.fix_event(metadata)
                self.stats['fixed_metadata'] += 1
            
            # 评估对局质量
            metadata, quality = self.evaluate_game_quality(metadata, file_path)
            
            # 质量过滤
            if quality < self.min_quality:
                logger.debug(f"质量不足的文件 ({quality} < {self.min_quality}): {file_path}")
                return None
            
            # 创建规范化的文件名
            new_filename = self.create_filename(metadata)
            
            # 根据组织方式决定目标路径
            if self.organize_by == 'quality':
                quality_bracket = f"{(quality // 10) * 10}-{(quality // 10) * 10 + 9}"
                target_dir = self.output_dir / f"Quality_{quality_bracket}"
            elif self.organize_by == 'player':
                # 寻找最著名的棋手
                player_names = [metadata.get('black_player', ''), metadata.get('white_player', '')]
                famous_player = None
                for name in player_names:
                    name_lower = name.lower()
                    for key, player_data in PLAYER_DB.items():
                        if name_lower in player_data['aliases'] or key in name_lower:
                            famous_player = player_data['name']
                            break
                    if famous_player:
                        break
                
                if famous_player:
                    target_dir = self.output_dir / f"Player_{famous_player}"
                else:
                    target_dir = self.output_dir / "Player_Other"
            elif self.organize_by == 'date':
                # 按年份组织
                try:
                    year = metadata.get('date', '').split('-')[0]
                    if year and year.isdigit():
                        target_dir = self.output_dir / f"Year_{year}"
                    else:
                        target_dir = self.output_dir / "Year_Unknown"
                except (IndexError, AttributeError):
                    target_dir = self.output_dir / "Year_Unknown"
            elif self.organize_by == 'event':
                # 按赛事组织
                event = metadata.get('event', '')
                if event:
                    # 确保目录名合法
                    event_dir = re.sub(r'[\\/*?:"<>|]', '_', event)
                    target_dir = self.output_dir / f"Event_{event_dir}"
                else:
                    target_dir = self.output_dir / "Event_Unknown"
            else:
                # 默认按质量分类
                quality_bracket = f"{(quality // 10) * 10}-{(quality // 10) * 10 + 9}"
                target_dir = self.output_dir / f"Quality_{quality_bracket}"
            
            # 创建目标目录
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / new_filename
            
            # 写入修复后的SGF内容
            if self.fix_metadata:
                # 更新SGF内容中的元数据
                updated_content = content
                for key, value in metadata.items():
                    if key in ['black_player', 'white_player', 'black_rank', 'white_rank', 
                              'event', 'date', 'result'] and value:
                        # 构建SGF属性标识
                        sgf_key = {'black_player': 'PB', 'white_player': 'PW', 
                                  'black_rank': 'BR', 'white_rank': 'WR', 
                                  'event': 'EV', 'date': 'DT', 'result': 'RE'}
                        
                        if key in sgf_key:
                            # 如果原始内容中存在该属性，则替换
                            pattern = f"{sgf_key[key]}\\[[^\\]]*\\]"
                            replacement = f"{sgf_key[key]}[{value}]"
                            if re.search(pattern, updated_content):
                                updated_content = re.sub(pattern, replacement, updated_content)
                            # 否则添加到开头
                            elif not re.search(f"{sgf_key[key]}\\[", updated_content):
                                # 在第一个分号之后添加
                                updated_content = re.sub(r'\(;', f'(;{sgf_key[key]}[{value}]', updated_content, count=1)
                
                with open(target_path, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
            else:
                # 直接复制原文件
                shutil.copy2(file_path, target_path)
            
            logger.debug(f"处理文件: {file_path} -> {target_path}")
            return target_path
        except Exception as e:
            logger.error(f"处理文件时出错 {file_path}: {e}")
            return None
    
    def run(self):
        """运行整理器"""
        logger.info(f"开始整理Gogames数据集，输入目录: {self.input_dir}, 输出目录: {self.output_dir}")
        
        # 查找所有SGF文件
        sgf_files = self.find_sgf_files()
        if not sgf_files:
            logger.warning("未找到SGF文件，退出")
            return
        
        # 显示目录分布
        logger.info("数据目录分布:")
        for directory, count in sorted(self.stats['directory_distribution'].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {directory}: {count} 个文件")
        
        # 使用线程池并行处理
        from tqdm import tqdm
        processed_files = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(self.process_sgf_file, file_path): file_path for file_path in sgf_files}
            
            for future in tqdm(concurrent.futures.as_completed(futures), 
                               total=len(futures), desc="处理SGF文件"):
                file_path = futures[future]
                try:
                    result = future.result()
                    if result:
                        processed_files.append(result)
                except Exception as e:
                    logger.error(f"处理文件时出错 {file_path}: {e}")
        
        # 保存处理统计信息
        self.save_stats()
        
        logger.info(f"Gogames数据集整理完成，共处理 {self.stats['valid_sgf_files']} 个有效文件，生成 {len(processed_files)} 个高质量文件")
        logger.info(f"已保存到: {self.output_dir}")
    
    def save_stats(self):
        """保存处理统计信息"""
        stats_file = self.output_dir / "processing_stats.json"
        try:
            # 转换defaultdict为普通字典
            stats_dict = {k: (dict(v) if isinstance(v, defaultdict) else v) for k, v in self.stats.items()}
            
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats_dict, f, ensure_ascii=False, indent=2)
            
            logger.info(f"处理统计信息已保存到: {stats_file}")
            
            # 生成简单的HTML报告
            self.generate_html_report()
        except Exception as e:
            logger.error(f"保存统计信息时出错: {e}")
    
    def generate_html_report(self):
        """生成HTML统计报告"""
        html_file = self.output_dir / "report.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Gogames数据集处理报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                h1, h2, h3 {{ color: #333; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .stats-container {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 30px; }}
                .stat-box {{ background: #f5f5f5; border-radius: 5px; padding: 15px; flex: 1; min-width: 200px; }}
                .stat-number {{ font-size: 24px; font-weight: bold; color: #0066cc; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .chart {{ width: 100%; height: 300px; margin-bottom: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Gogames数据集处理报告</h1>
                <p>处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="stats-container">
                    <div class="stat-box">
                        <h3>总文件数</h3>
                        <div class="stat-number">{self.stats['total_files']}</div>
                    </div>
                    <div class="stat-box">
                        <h3>有效SGF文件</h3>
                        <div class="stat-number">{self.stats['valid_sgf_files']}</div>
                    </div>
                    <div class="stat-box">
                        <h3>高质量文件</h3>
                        <div class="stat-number">{self.stats['high_quality']}</div>
                    </div>
                    <div class="stat-box">
                        <h3>重复文件</h3>
                        <div class="stat-number">{self.stats['duplicates']}</div>
                    </div>
                </div>
                
                <h2>棋谱质量分布</h2>
                <table>
                    <tr>
                        <th>质量分数</th>
                        <th>文件数量</th>
                        <th>百分比</th>
                    </tr>
        """
        
        # 添加质量分布表格
        quality_brackets = sorted(self.stats['games_by_quality'].keys())
        for bracket in quality_brackets:
            count = self.stats['games_by_quality'][bracket]
            percentage = count / self.stats['valid_sgf_files'] * 100 if self.stats['valid_sgf_files'] > 0 else 0
            html_content += f"""
                    <tr>
                        <td>{bracket}-{bracket+9}</td>
                        <td>{count}</td>
                        <td>{percentage:.2f}%</td>
                    </tr>
            """
        
        html_content += """
                </table>
                
                <h2>顶级棋手出场次数</h2>
                <table>
                    <tr>
                        <th>棋手</th>
                        <th>出场次数</th>
                    </tr>
        """
        
        # 添加棋手出场次数表格
        top_players = sorted(self.stats['games_by_player'].items(), key=lambda x: x[1], reverse=True)[:20]
        for player, count in top_players:
            html_content += f"""
                    <tr>
                        <td>{player}</td>
                        <td>{count}</td>
                    </tr>
            """
        
        html_content += """
                </table>
                
                <h2>赛事分布</h2>
                <table>
                    <tr>
                        <th>赛事</th>
                        <th>对局数量</th>
                    </tr>
        """
        
        # 添加赛事分布表格
        top_events = sorted(self.stats['games_by_event'].items(), key=lambda x: x[1], reverse=True)[:20]
        for event, count in top_events:
            html_content += f"""
                    <tr>
                        <td>{event}</td>
                        <td>{count}</td>
                    </tr>
            """
        
        html_content += """
                </table>
                
                <h2>年份分布</h2>
                <table>
                    <tr>
                        <th>年份</th>
                        <th>对局数量</th>
                    </tr>
        """
        
        # 添加年份分布表格
        years = sorted(self.stats['games_by_year'].items(), key=lambda x: x[0])
        for year, count in years:
            html_content += f"""
                    <tr>
                        <td>{year}</td>
                        <td>{count}</td>
                    </tr>
            """
        
        html_content += """
                </table>
                
                <h2>目录分布</h2>
                <table>
                    <tr>
                        <th>目录</th>
                        <th>文件数量</th>
                    </tr>
        """
        
        # 添加目录分布表格
        directories = sorted(self.stats['directory_distribution'].items(), key=lambda x: x[1], reverse=True)
        for directory, count in directories:
            html_content += f"""
                    <tr>
                        <td>{directory}</td>
                        <td>{count}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML报告已保存到: {html_file}")


def main():
    parser = argparse.ArgumentParser(description="Gogames数据集整理工具")
    parser.add_argument("input", help="输入目录，包含杂乱的SGF文件")
    parser.add_argument("output", help="输出目录，用于保存整理后的文件")
    parser.add_argument("--quality", "-q", type=int, default=70, help="最低质量阈值(0-100)，默认:70")
    parser.add_argument("--organize", "-o", choices=['quality', 'player', 'date', 'event'], 
                       default='quality', help="组织方式，默认按质量分类")
    parser.add_argument("--no-fix", action="store_false", dest="fix_metadata", 
                       help="不修复和标准化元数据")
    parser.add_argument("--no-path-extract", action="store_false", dest="extract_from_path", 
                       help="不从文件路径提取元数据")
    parser.add_argument("--keep-dupes", action="store_false", dest="remove_duplicates", 
                       help="保留重复文件")
    parser.add_argument("--workers", "-w", type=int, default=4, 
                       help="并行处理的工作线程数，默认:4")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细日志")
    args = parser.parse_args()
    
    organizer = GogamesOrganizer(
        input_dir=args.input,
        output_dir=args.output,
        min_quality=args.quality,
        organize_by=args.organize,
        fix_metadata=args.fix_metadata,
        extract_from_path=args.extract_from_path,
        remove_duplicates=args.remove_duplicates,
        workers=args.workers,
        verbose=args.verbose
    )
    
    try:
        organizer.run()
        return 0
    except KeyboardInterrupt:
        logger.info("用户中断，停止处理")
        organizer.save_stats()
        return 1
    except Exception as e:
        logger.error(f"处理时出错: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())