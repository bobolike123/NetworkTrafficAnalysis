import pickle
rankNUM=25
Nor=11709971
BF=274938
WA=40791
IF=148956
BN=18168
PS=1604317
def cal_ratio():
    All=Nor+BF+WA+IF+BN+PS
    print(f'the percentage ratio of Nor:{Nor/All}, BF:{BF/All}, WA:{WA/All}, IF: {IF/All}, BN: {BN/All}, PS:{PS/All}')
# the percentage ratio of Nor:0.8487244567552075, BF:0.019927171868432743, WA:0.00295648207117692, IF: 0.010796149724062398, BN: 0.0013167945446089157, PS:0.11627894503651155
if __name__ == '__main__':
    cal_ratio()

