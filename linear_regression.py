import numpy as np
import tensorflow as tf

# Model parameters
W = tf.Variable([[0.0], [0.0]], tf.float32) # [0.417362099], [5.216590809]   [[0.3], [4.5]]
b = tf.Variable([0.0], tf.float32) #[77.98253861]       [69.0]

tf.summary.histogram('b', b)
tf.summary.histogram('W', W)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = tf.matmul(x, W) + b
y = tf.placeholder(tf.float32)
# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
tf.summary.scalar('loss', loss)
# optimizer
train = tf.train.GradientDescentOptimizer(0.000001).minimize(loss)
# training data
x_train = [[84.0, 46.0], [73, 20], [65, 52], [70, 30], [76, 57], [69, 25], [63, 28], [72, 36], [79, 57], [75, 44],
              [27, 24], [89, 31], [65, 52], [57, 23], [59, 60], [69, 48], [60, 34], [79, 51], [75, 50], [82, 34],
              [59, 46], [67, 23], [85, 37], [55, 40], [63, 30]]
y_train = [[354], [190], [405], [263], [451], [302], [288], [385], [402], [365], [209], [290], [346], [254], [395], [434], [220], [374], [308], [220], [311], [181], [274], [303], [244]]
#y_train = [353.004132118, 212.7817880027, 376.3737970937, 263.6956097942, 407.0477342244, 237.1952936515, 250.3408934851, 295.8298788446, 408.2998205207, 338.8146916113, 214.4494946933, 276.8420804801, 376.3737970937, 221.7537668484, 415.6023509712, 357.1768822537, 280.3883520415, 377.000275668, 370.114236464, 289.5703182149, 342.5700796482, 225.9273878363, 306.4721769377, 309.6010864002, 260.7740751027]
#y_train = [[353.004132118], [212.7817880027], [376.3737970937], [263.6956097942], [407.0477342244], [237.1952936515], [250.3408934851], [295.8298788446], [408.2998205207], [338.8146916113], [214.4494946933], [276.8420804801], [376.3737970937], [221.7537668484], [415.6023509712], [357.1768822537], [280.3883520415], [377.000275668], [370.114236464], [289.5703182149], [342.5700796482], [225.9273878363], [306.4721769377], [309.6010864002], [260.7740751027]]


# training loop
init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init) # reset values to wrong
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./" ,sess.graph)



for i in range(3000):
#  summary, _ = sess.run([merged, train], feed_dict={x:x_train, y:y_train})
  curr_W, curr_b, curr_loss, summary, _ = sess.run([W, b, loss, merged, train], feed_dict={x:x_train, y:y_train})
  print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
  writer.add_summary(summary, global_step=i)



# open cmd comand fenster
# zuerst auf ordner indem event datei liegt wechseln
# tensorboard --logdir=. --debug
# you can leave everything open, it will reload new eventfile every 120 sec
