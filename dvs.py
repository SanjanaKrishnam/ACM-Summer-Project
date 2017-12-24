import cPickle, gzip,numpy as np
from numpy import random
import theano
from theano import shared,function,tensor as T

# Load the dataset

f = gzip.open('/Users/Sanjana/Desktop/samples/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

#Structure of the neural network

inputlayersize=784
hiddenlayersize1=590
hiddenlayersize2=203
num_labels=10

train_image=list(train_set[0])
train_label=np.reshape(np.array(train_set[1]),(50000,1))
valid_image=list(valid_set[0])
valid_label=np.reshape(np.array(valid_set[1]),(10000,1))
test_image=list(valid_set[0])
test_label=np.reshape(np.array(valid_set[1]),(10000,1))
train_labelm = (np.eye(num_labels)[list(train_set[1]), :]).tolist()
valid_labelm=(np.eye(num_labels)[list(valid_set[1]), :]).tolist()
test_labelm=(np.eye(num_labels)[list(test_set[1]), :]).tolist()

lamdas=[0,0.0001,0.001,0.01,0.1,1]
max=0

for lamda in lamdas:

    #Declare parameters

    alpha=0.2


    #Initialize weights

    rng = random.RandomState(1234)

    def layer(n_in, n_out):
        return theano.shared(value=np.asarray(rng.uniform(low=-1.0, high=1.0, size=(n_in, n_out)), dtype=theano.config.floatX), name='W',borrow=True)


    theta1 = layer(inputlayersize + 1, hiddenlayersize1)
    theta2 = layer(hiddenlayersize1 + 1, hiddenlayersize2)
    theta3 = layer(hiddenlayersize2 + 1, num_labels)


    #finding cost


    x=T.dmatrix('x')
    y=T.dmatrix('y')

    m=x.shape[0]

    a1=T.concatenate([T.ones((m,1)),x],axis=1)
    z2=T.dot(a1,theta1)
    a2=T.concatenate([T.ones((m,1)),T.nnet.sigmoid(z2)],axis=1)
    z3=T.dot(a2,theta2)
    a3=T.concatenate([T.ones((m,1)),T.nnet.sigmoid(z3)],axis=1)
    h_theta=T.nnet.sigmoid(T.dot(a3,theta3))
    a4=h_theta

    xent = -((y * T.log(a4) + (1 - y) * T.log(1 - a4)).sum())
    regularization = float(lamda) / 2 * ((theta1[1:,:] ** 2).sum() + (theta2[1:,:] ** 2).sum()+theta3[1:,:] ** 2).sum()
    cost=(xent+regularization)/m

    predicted = T.argmax(a4, axis=1)
    predicted = T.reshape(predicted, (m, 1))

    D1=T.grad(cost,theta1)
    D2=T.grad(cost,theta2)
    D3=T.grad(cost,theta3)


    f = function(inputs=[x,y],outputs=cost,updates=((theta1, theta1 - alpha * D1), (theta2, theta2 - alpha * D2),(theta3, theta3 - alpha * D3)))
    g = function(inputs=[x,y],outputs=[cost,predicted])

    for i in range(1500):
        if i % 100 == 0:
            alpha = float(alpha) / 10
        print(f(train_image,train_labelm))

    print ("for lamda:")
    print(lamda)

    cost,pred=g(train_image,train_labelm)
    print("Cost of train set:")
    print(cost)
    print("Accuracy of train set:")
    print(((pred == train_label) * 1).mean() * 100)


    cost,pred=g(valid_image,valid_labelm)
    print("Cost of cross valid set:")
    print(cost)
    print("Accuracy of cross valid set:")
    accuracy=((pred == valid_label) * 1).mean() * 100
    if accuracy>max:
        lamdafinal=lamda
        thetafinal1=theta1
        thetafinal2 = theta2
        thetafinal3 = theta3
    print(accuracy)


    i= str(lamda)+'parameter.save'
    f = open(i, 'wb')
    for obj in [theta1, theta2, theta3]:
       cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


lamda=lamdafinal
theta1=thetafinal1
theta2=thetafinal2
theta3=thetafinal3

print("final lamda:")
print(lamda)
cost,pred=g(test_image,test_labelm)
print("Cost of test set:")
print(cost)
print("Accuracy of test set:")
print(((pred == test_label) * 1).mean() * 100)

