import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('/tmp/data', one_hot=True)

n_node_hl1=550
n_node_hl2=450
n_node_hl3=500
n_node_hl4=500
n_node_hl5=500

n_classes=10
batch_size=128


x=tf.placeholder('float',[None, 784])
y=tf.placeholder('float')

def neural_model(data):
    hidden1l={'weights':tf.Variable(tf.random_normal([784,n_node_hl1])), 'biases': tf.Variable(tf.random_normal([n_node_hl1]))}
    hidden2l={'weights':tf.Variable(tf.random_normal([n_node_hl1, n_node_hl2])), 'biases': tf.Variable(tf.random_normal([n_node_hl2]))}
    hidden3l={'weights':tf.Variable(tf.random_normal([n_node_hl2, n_node_hl3])), 'biases': tf.Variable(tf.random_normal([n_node_hl3]))}
    hidden4l={'weights':tf.Variable(tf.random_normal([n_node_hl3, n_node_hl4])), 'biases': tf.Variable(tf.random_normal([n_node_hl4]))}
    hidden5l={'weights':tf.Variable(tf.random_normal([n_node_hl4, n_node_hl5])), 'biases': tf.Variable(tf.random_normal([n_node_hl5]))}
    
    output_layer={'weights':tf.Variable(tf.random_normal([n_node_hl5,  n_classes])), 'biases': tf.Variable(tf.random_normal([n_classes]))}
    
    l1=tf.add(tf.matmul(data, hidden1l['weights']),hidden1l['biases'])
    l1=tf.nn.relu(l1)
     
    l2=tf.add(tf.matmul(l1, hidden2l['weights']),hidden2l['biases'])
    l2=tf.nn.relu(l2)
     
    l3=tf.add(tf.matmul(l2, hidden3l['weights']),hidden3l['biases'])
    l3=tf.nn.relu(l3)
    
    l4=tf.add(tf.matmul(l3, hidden4l['weights']),hidden4l['biases'])
    l4=tf.nn.relu(l4)
    
    l5=tf.add(tf.matmul(l4, hidden5l['weights']),hidden4l['biases'])
    l5=tf.nn.relu(l5)
     
    output=tf.add(tf.matmul(l5, output_layer['weights']),output_layer['biases'])
    return output
 
    
def train_neural(x):
    prediction=neural_model(x)
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    
    optimizer=tf.train.AdamOptimizer().minimize(cost)
    hm_epochs=10
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(hm_epochs):
            epoch_loss=0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x,epoch_y=mnist.train.next_batch(batch_size)
                _, c=sess.run([optimizer,cost], feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss+=c
            print('epoch', epoch, 'completed out og', hm_epochs,'loss:',epoch_loss)
        correct=tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        
        accuracy=tf.reduce_mean(tf.cast(correct,'float'))
        print('accuracy', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
    
     
train_neural(x)
