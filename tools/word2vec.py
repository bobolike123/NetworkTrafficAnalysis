#代码不全，但是舍不得删
def create_indices():
    each_flow_data = b''

    txt4 = b'\x00\xe0\xb1\x87\xf5\x94\x00\x01\x02r\xabU\x08\x00E\x00\x05\xdcN\r@\x004\x06m\xa8mHU\x05\xc0\xa8\x02q\x00P\x08\x03\x1cO\xc3j\x04\x96\r!P\x10\x16\xd0U\x9f\x00\x00'
    txt5 = b'\x00\xe0\xb1\x87\xf5\x94\x00\x01\x02r\xabU\x08\x00E\x00\x05\xdcN\x0f@\x004\x06m\xa6mHU\x05\xc0\xa8\x02q\x00P\x08\x03\x1cO\xce\xd2\x04\x96\r!P\x10\x16\xd0$\x91\x00\x00'

    notFirstUse = check_feature_dict()  # 判断是否第一次运行程序,值为False表示第一次用，为True则不是

    txt_list = list()
    txt_list.append(txt4)
    txt_list.append(txt5)

    for eachPcaket in txt_list:
        print(eachPcaket)
        each_flow_data += eachPcaket
    chars = set(each_flow_data)
    print('total chars:', len(chars))
    print(chars)

    if not notFirstUse:  # 如果第一次运行，则新建字典文件
        char_indices = dict((c, i) for i, c in enumerate(chars))
        with open('./feature_dict.pkl', 'wb') as f:
            pickle.dump(char_indices, f)
            print('feature_dict has been created')
    else:  # 非第一次用，则载入原字典，检查字典需不需要更新
        with open('./feature_dict.pkl', 'rb') as f:
            feature_dict = pickle.load(f)
            print('feature_dict has been loaded')
            print('old feature_dict:', feature_dict)

        fd_len = len(feature_dict)
        update_dict = dict()
        for char in chars:
            # 如果输入的数据中有字典里面没有的键，则在字典末尾追加新的键和键值，新健值由字典长度决定
            if not feature_dict.__contains__(char):  # py2.x是dict.has_key(key),而py3.x则以dict.__contains__(key)替代
                update_dict[char] = fd_len
                fd_len += 1
        print('update_dict:', update_dict)
        feature_dict.update(update_dict)
        print('new feature_dict:', feature_dict)

        with open('./feature_dict.pkl', 'wb') as f2:
            pickle.dump(feature_dict, f2)
            if len(update_dict):  # 如果有更新：
                print('feature_dict has been updated!')
