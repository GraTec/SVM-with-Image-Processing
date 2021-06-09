import matplotlib.pyplot as plt


y_time = [[10.24061595, 10.136285849999997, 9.692289704000004, 10.152323445999997, 10.023762051999995], [10.438174830999998, 10.328865231000009, 10.238712541000012,
                                                                                                         10.566841119999992, 10.610555854000012], [10.241711877, 10.376887475000018, 10.052556737999964,
                                                                                                                                                   10.490463972999976, 10.358932833999972], [9.613820408000038, 9.882703816999992, 9.614343541999972,
                                                                                                                                                                                             9.833416554999985, 10.051817602000028], [9.354568711000013, 9.299844808999978, 9.228252970000028, 9.341733488999921,
                                                                                                                                                                                                                                      9.51834798599998]]
y_error = [[0.9, 0.9, 0.9, 0.9, 0.9], [0.9, 0.9, 0.9, 0.9, 0.9], [0.9, 0.9, 0.9001, 0.9, 0.9], [
    0.9001, 0.9, 0.9002, 0.9, 0.9002], [0.9003, 0.9002, 0.9007000000000001, 0.9003, 0.9005]]

yn_time = [9.46283037, 9.386277245999995,
           9.179519941999999, 9.378836269999994, 9.39051176000001]
yn_error = [0.9003, 0.9002, 0.9009, 0.9003, 0.9004]


def myplot(y_time, k):
    x = range(len(y_time))
    plt.plot(x, y_time, marker='o', label='k='+str(k))

    # plt.show()


klist = [1, 2, 4, 8, 16]

# for i in range(5):
#     myplot(y_time[i], klist[i])

# plt.xlabel('Training Dataset')
# plt.ylabel('Times/s')
# plt.axis([-1, 5, 7, 12])
# plt.legend()
# plt.savefig("./image/svmwithSVD_time.png")

# for i in range(5):
#     myplot(y_error[i], klist[i])

# plt.xlabel('Training Dataset')
# plt.ylabel('Accuracy')
# # plt.axis([-1, 5, 0.88, 0.92])
# plt.legend()
# plt.savefig("./image/svmwithSVD_accuracy.png")


# for i in range(5):
#     plt.plot(range(len(yn_time)), yn_time, marker='o', label='NormalSVM')
#     myplot(y_time[i], klist[i])

#     plt.xlabel('Training Dataset')
#     plt.ylabel('Times/s')
#     # plt.axis([-1, 5, 0.88, 0.92])
#     plt.legend()
#     plt.savefig('./image/svm_time'+str(i)+'.png')
#     plt.cla()

for i in range(5):
    plt.plot(range(len(yn_error)), yn_error, marker='o', label='NormalSVM')
    myplot(y_error[i], klist[i])

    plt.xlabel('Training Dataset')
    plt.ylabel('Accuracy')
    # plt.axis([-1, 5, 0.88, 0.92])
    plt.legend()
    plt.savefig('./image/svm_accuracy'+str(i)+'.png')
    plt.cla()
