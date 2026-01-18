import cpca
import pandas as pd

# 英文城市名到中文城市名的映射
CITY_EN_TO_CN = {
    # 直辖市
    "beijing": "北京市",
    "shanghai": "上海市",
    "tianjin": "天津市",
    "chongqing": "重庆市",
    # 广东省
    "guangzhou": "广州市",
    "shenzhen": "深圳市",
    "dongguan": "东莞市",
    "foshan": "佛山市",
    "zhuhai": "珠海市",
    "zhongshan": "中山市",
    "huizhou": "惠州市",
    "jiangmen": "江门市",
    "zhanjiang": "湛江市",
    "shantou": "汕头市",
    "shaoguan": "韶关市",
    "zhaoqing": "肇庆市",
    "maoming": "茂名市",
    "meizhou": "梅州市",
    "shanwei": "汕尾市",
    "yangjiang": "阳江市",
    "qingyuan": "清远市",
    "chaozhou": "潮州市",
    "jieyang": "揭阳市",
    "yunfu": "云浮市",
    "heyuan": "河源市",
    # 江苏省
    "nanjing": "南京市",
    "suzhou": "苏州市",
    "wuxi": "无锡市",
    "changzhou": "常州市",
    "nantong": "南通市",
    "xuzhou": "徐州市",
    "yangzhou": "扬州市",
    "zhenjiang": "镇江市",
    "taizhou_js": "泰州市",  # 江苏泰州
    "yancheng": "盐城市",
    "lianyungang": "连云港市",
    "huaian": "淮安市",
    "suqian": "宿迁市",
    # 浙江省
    "hangzhou": "杭州市",
    "ningbo": "宁波市",
    "wenzhou": "温州市",
    "jiaxing": "嘉兴市",
    "huzhou": "湖州市",
    "shaoxing": "绍兴市",
    "jinhua": "金华市",
    "quzhou": "衢州市",
    "zhoushan": "舟山市",
    "taizhou": "台州市",  # 浙江台州 (默认)
    "lishui": "丽水市",
    # 山东省
    "jinan": "济南市",
    "qingdao": "青岛市",
    "zibo": "淄博市",
    "zaozhuang": "枣庄市",
    "dongying": "东营市",
    "yantai": "烟台市",
    "weifang": "潍坊市",
    "jining": "济宁市",
    "taian": "泰安市",
    "weihai": "威海市",
    "rizhao": "日照市",
    "linyi": "临沂市",
    "dezhou": "德州市",
    "liaocheng": "聊城市",
    "binzhou": "滨州市",
    "heze": "菏泽市",
    # 河北省
    "shijiazhuang": "石家庄市",
    "tangshan": "唐山市",
    "qinhuangdao": "秦皇岛市",
    "handan": "邯郸市",
    "xingtai": "邢台市",
    "baoding": "保定市",
    "zhangjiakou": "张家口市",
    "chengde": "承德市",
    "cangzhou": "沧州市",
    "langfang": "廊坊市",
    "hengshui": "衡水市",
    # 河南省
    "zhengzhou": "郑州市",
    "kaifeng": "开封市",
    "luoyang": "洛阳市",
    "pingdingshan": "平顶山市",
    "anyang": "安阳市",
    "hebi": "鹤壁市",
    "xinxiang": "新乡市",
    "jiaozuo": "焦作市",
    "puyang": "濮阳市",
    "xuchang": "许昌市",
    "luohe": "漯河市",
    "sanmenxia": "三门峡市",
    "nanyang": "南阳市",
    "shangqiu": "商丘市",
    "xinyang": "信阳市",
    "zhoukou": "周口市",
    "zhumadian": "驻马店市",
    "jiyuan": "济源市",
    # 湖北省
    "wuhan": "武汉市",
    "huangshi": "黄石市",
    "shiyan": "十堰市",
    "yichang": "宜昌市",
    "xiangyang": "襄阳市",
    "ezhou": "鄂州市",
    "jingmen": "荆门市",
    "xiaogan": "孝感市",
    "jingzhou": "荆州市",
    "huanggang": "黄冈市",
    "xianning": "咸宁市",
    "suizhou": "随州市",
    # 湖南省
    "changsha": "长沙市",
    "zhuzhou": "株洲市",
    "xiangtan": "湘潭市",
    "hengyang": "衡阳市",
    "shaoyang": "邵阳市",
    "yueyang": "岳阳市",
    "changde": "常德市",
    "zhangjiajie": "张家界市",
    "yiyang": "益阳市",
    "chenzhou": "郴州市",
    "yongzhou": "永州市",
    "huaihua": "怀化市",
    "loudi": "娄底市",
    # 四川省
    "chengdu": "成都市",
    "zigong": "自贡市",
    "panzhihua": "攀枝花市",
    "luzhou": "泸州市",
    "deyang": "德阳市",
    "mianyang": "绵阳市",
    "guangyuan": "广元市",
    "suining": "遂宁市",
    "neijiang": "内江市",
    "leshan": "乐山市",
    "nanchong": "南充市",
    "meishan": "眉山市",
    "yibin": "宜宾市",
    "guangan": "广安市",
    "dazhou": "达州市",
    "yaan": "雅安市",
    "bazhong": "巴中市",
    "ziyang": "资阳市",
    # 陕西省
    "xian": "西安市",
    "tongchuan": "铜川市",
    "baoji": "宝鸡市",
    "xianyang": "咸阳市",
    "weinan": "渭南市",
    "yanan": "延安市",
    "hanzhong": "汉中市",
    "yulin": "榆林市",
    "ankang": "安康市",
    "shangluo": "商洛市",
    # 辽宁省
    "shenyang": "沈阳市",
    "dalian": "大连市",
    "anshan": "鞍山市",
    "fushun": "抚顺市",
    "benxi": "本溪市",
    "dandong": "丹东市",
    "jinzhou": "锦州市",
    "yingkou": "营口市",
    "fuxin": "阜新市",
    "liaoyang": "辽阳市",
    "panjin": "盘锦市",
    "tieling": "铁岭市",
    "chaoyang": "朝阳市",
    "huludao": "葫芦岛市",
    # 吉林省
    "changchun": "长春市",
    "jilin": "吉林市",
    "siping": "四平市",
    "liaoyuan": "辽源市",
    "tonghua": "通化市",
    "baishan": "白山市",
    "songyuan": "松原市",
    "baicheng": "白城市",
    # 黑龙江省
    "harbin": "哈尔滨市",
    "qiqihar": "齐齐哈尔市",
    "jixi": "鸡西市",
    "hegang": "鹤岗市",
    "shuangyashan": "双鸭山市",
    "daqing": "大庆市",
    "yichun_hlj": "伊春市",  # 黑龙江伊春
    "jiamusi": "佳木斯市",
    "qitaihe": "七台河市",
    "mudanjiang": "牡丹江市",
    "heihe": "黑河市",
    "suihua": "绥化市",
    # 安徽省
    "hefei": "合肥市",
    "wuhu": "芜湖市",
    "bengbu": "蚌埠市",
    "huainan": "淮南市",
    "maanshan": "马鞍山市",
    "huaibei": "淮北市",
    "tongling": "铜陵市",
    "anqing": "安庆市",
    "huangshan": "黄山市",
    "chuzhou": "滁州市",
    "fuyang": "阜阳市",
    "suzhou_ah": "宿州市",  # 安徽宿州
    "luan": "六安市",
    "bozhou": "亳州市",
    "chizhou": "池州市",
    "xuancheng": "宣城市",
    # 福建省
    "fuzhou": "福州市",
    "xiamen": "厦门市",
    "putian": "莆田市",
    "sanming": "三明市",
    "quanzhou": "泉州市",
    "zhangzhou": "漳州市",
    "nanping": "南平市",
    "longyan": "龙岩市",
    "ningde": "宁德市",
    # 江西省
    "nanchang": "南昌市",
    "jingdezhen": "景德镇市",
    "pingxiang": "萍乡市",
    "jiujiang": "九江市",
    "xinyu": "新余市",
    "yingtan": "鹰潭市",
    "ganzhou": "赣州市",
    "jian": "吉安市",
    "yichun_jx": "宜春市",  # 江西宜春
    "fuzhou_jx": "抚州市",  # 江西抚州
    "shangrao": "上饶市",
    # 山西省
    "taiyuan": "太原市",
    "datong": "大同市",
    "yangquan": "阳泉市",
    "changzhi": "长治市",
    "jincheng": "晋城市",
    "shuozhou": "朔州市",
    "jinzhong": "晋中市",
    "yuncheng": "运城市",
    "xinzhou": "忻州市",
    "linfen": "临汾市",
    "lvliang": "吕梁市",
    # 云南省
    "kunming": "昆明市",
    "qujing": "曲靖市",
    "yuxi": "玉溪市",
    "baoshan": "保山市",
    "zhaotong": "昭通市",
    "lijiang": "丽江市",
    "puer": "普洱市",
    "lincang": "临沧市",
    # 贵州省
    "guiyang": "贵阳市",
    "liupanshui": "六盘水市",
    "zunyi": "遵义市",
    "anshun": "安顺市",
    "bijie": "毕节市",
    "tongren": "铜仁市",
    # 甘肃省
    "lanzhou": "兰州市",
    "jiayuguan": "嘉峪关市",
    "jinchang": "金昌市",
    "baiyin": "白银市",
    "tianshui": "天水市",
    "wuwei": "武威市",
    "zhangye": "张掖市",
    "pingliang": "平凉市",
    "jiuquan": "酒泉市",
    "qingyang": "庆阳市",
    "dingxi": "定西市",
    "longnan": "陇南市",
    # 其他省份主要城市
    "haikou": "海口市",
    "sanya": "三亚市",
    "nanning": "南宁市",
    "liuzhou": "柳州市",
    "guilin": "桂林市",
    "urumqi": "乌鲁木齐市",
    "korla": "库尔勒市",
    "lhasa": "拉萨市",
    "yinchuan": "银川市",
    "xining": "西宁市",
    "hohhot": "呼和浩特市",
    "baotou": "包头市",
    # 香港、澳门、台湾
    "hongkong": "香港",
    "hong kong": "香港",
    "macau": "澳门",
    "macao": "澳门",
    "taipei": "台北市",
    "kaohsiung": "高雄市",
    "taichung": "台中市",
}


def get_factor(city_name, gef_path):
    """获取城市对应的电网排放因子"""
    province = get_province_from_city(city_name)
    if province is None:
        raise ValueError(f"Unknown city: {city_name}")
    factor = fetch_china(province=province, gef_path=gef_path)
    print(f"CO2 emission factor for {city_name} ({province}): {factor} gCO2eq/kWh")
    return factor


def fetch_china(province, gef_path):
    """从CSV文件获取省份的电网排放因子"""
    data = pd.read_csv(gef_path, encoding="gbk")
    eg_province = data.loc[data["城市"] == province, "因子"].values[0]
    return float(eg_province)


def get_province_from_city(city_name_en: str) -> str:
    """
    根据城市英文名称获取省级行政区名称 (使用 cpca 库)

    参数:
        city_name_en: 城市英文名称

    返回:
        省级行政区名称（中文），如 "广东"、"北京" 等
    """
    city = city_name_en.lower().strip().split(",")[0].replace("'", "")

    # 先转换为中文城市名
    city_cn = CITY_EN_TO_CN.get(city)
    if city_cn is None:
        return None

    # 使用 cpca 提取省份信息
    df = cpca.transform([city_cn])
    province = df.iloc[0]["省"]

    # cpca 返回的省份名称可能带"省"后缀，需要去掉以匹配 gef 文件
    if province and province.endswith("省"):
        province = province[:-1]

    return province
