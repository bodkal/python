import numpy as np
import matplotlib.pyplot as plt


if __name__=='__main__':
    index=3 # 5  7 9

    row_data=np.loadtxt(f'test_{index.__str__()}_agent1.csv',delimiter=',',skiprows = 1,dtype=str)
    data_joint=np.zeros(row_data[0].__len__()-index*2+1)
    for i,data in enumerate(row_data[:,1:index*2+1]):
        tmp_row_data=''.join(data)
        data_joint=np.vstack([data_joint,np.hstack([row_data[i,0],tmp_row_data,row_data[i,index*2+1:]])])

    data_joint=data_joint[1:]
    exp=(row_data.__len__())//4



    singlton_data=row_data[np.where(row_data[:,index*2+2]=='singlton' )]
    max_data=row_data[np.where(row_data[:,index*2+2]=='max')]
    mix_data=row_data[np.where(row_data[:,index*2+2]=='mix' )]
    mtsp_data=row_data[np.where(row_data[:,index*2+2]=='mtsp')]

    # bad_singlton=np.where(singlton_data[:, index * 2 + 1] == '-1')[0].tolist()
    # bad_max=np.where(max_data[:, index * 2 + 1] == '-1')[0].tolist()
    # bad_mix   = np.where(mix_data[:, index * 2 + 1] == '-1')[0].tolist()
    # bad_mtsp =  np.where(mtsp_data[:, index * 2 + 1] == '-1')[0].tolist()


    bad_singlton=np.where(singlton_data[:, index * 2 + 1].astype(float) > 300)[0].tolist()
    bad_max=np.where(max_data[:, index * 2 + 1].astype(float) > 300)[0].tolist()
    bad_mix   = np.where(mix_data[:, index * 2 + 1].astype(float) > 300)[0].tolist()
    bad_mtsp =  np.where(mtsp_data[:, index * 2 + 1].astype(float) > 300)[0].tolist()

    bad=bad_singlton+bad_max+bad_max+bad_mtsp


    sum_s=np.mean([singlton_data[i,index*2+1].astype(float) for i in range(exp) if  i not in bad])
    sum_max=np.mean([max_data[i,index*2+1].astype(float) for i in range(exp) if i not in bad])
    sum_min=np.mean([mix_data[i,index*2+1].astype(float) for i in range(exp) if i not in bad])
    sum_mtsp=np.mean([mtsp_data[i,index*2+1].astype(float) for i in range(exp) if i not in bad])
    plt.figure('sum')



    singlton_data=np.delete(singlton_data,bad,axis=0)
    max_data=np.delete(max_data,bad,axis=0)
    mix_data=np.delete(mix_data,bad,axis=0)
    mtsp_data=np.delete(mtsp_data,bad,axis=0)

    plt.bar(1, sum_s )
    plt.bar(2,sum_max)
    plt.bar(3,sum_min)
    plt.bar(4,sum_mtsp)

    plt.ylabel('time [sec]')
    #plt.hist([sum_s,sum_max,sum_min,sum_mtsp], label=['singlton','max', 'mtsp','mix'])
    plt.legend([f'singlton {(1-bad_singlton.__len__()/exp)*100} %',
                f'max {(1-bad_max.__len__()/exp)*100} %',
                f'mix {(1-bad_mix.__len__()/exp)*100} %',
                f'mtsp {(1-bad_mtsp.__len__()/exp)*100} %'])

    plt.figure('exp')
    plt.plot(range(singlton_data[:,index*2+1].__len__()),singlton_data[:,index*2+1].astype(float),'bo')
    plt.plot(range(max_data[:,index*2+1].__len__()),max_data[:,index*2+1].astype(float),'o',color='orange')
    plt.plot(range(mtsp_data[:,index*2+1].__len__()),mtsp_data[:,index*2+1].astype(float),'ro')
    plt.plot(range(mix_data[:,index*2+1].__len__()),mix_data[:,index*2+1].astype(float),'go')

    plt.ylabel('time')
    plt.legend([f'singlton {(1-bad_singlton.__len__()/exp)*100} %',
                f'max {(1-bad_max.__len__()/exp)*100} %',
                f'mix {(1-bad_mix.__len__()/exp)*100} %',
                f'mtsp {(1-bad_mtsp.__len__()/exp)*100} %'])
    plt.grid()

    plt.show()