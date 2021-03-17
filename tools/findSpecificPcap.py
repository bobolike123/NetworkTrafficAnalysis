# preproccess/findPcap.py的简化版
import glob
import os
import shutil

def set_goal_pcap(pcapname, sip, dip):
    sip = sip.replace('.', '-')
    protocol = "*"
    sport = "*"
    dip = dip.replace('.', '-')
    dport = "*"
    output = pcapname + '.' + protocol + '_' + sip + '_' + sport + '_' + dip + '_' + dport + '.pcap'
    print(output)
    return output

def glob_goal_pcap(fpath,filename):
    flist=glob.glob(os.path.join(fpath,filename))
    print(f"找到符合要求的数据包{len(flist)}个")
    return flist

def move_goal_pcap(flist,newDirName):
    dirpath=os.path.join(os.path.dirname(flist[0]),newDirName)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    for file in flist:
        shutil.move(file,os.path.join(dirpath,os.path.basename(file)))
    print('移动完毕')

def run():
    fname = set_goal_pcap("Friday-WorkingHours.pcap", "172.16.0.1", "192.168.10.50")
    flist=glob_goal_pcap(r"K:\dataset\CIC-IDS-2017\Port Scan\2_Session\AllLayers\Friday-WorkingHours-ALL", fname)
    move_goal_pcap(flist,"PortScanPcap")

if __name__ == "__main__":
    run()
