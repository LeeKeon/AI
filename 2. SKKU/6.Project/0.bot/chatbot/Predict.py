import tensorflow as tf

class Predict:
    def __init__(self, modelPath):
        print('Predict init Restore Model Path : ', modelPath)
        self.modelPath = modelPath
        self.saver = restore(self)

    def predict_result(self):
        predict = sess.run(Y_pred, feed_dict={X: testX})
        rmse_val = sess.run(rmse, feed_dict={
                    targets: testY, predictions: test_predict})
        return predict,rmse_val

    def restore(self, sess):
        print(' - Restoring variables...')
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        var_list = [var for var in tf.all_variables()]
        saver = tf.train.Saver(var_list)
        saver.restore(sess, modelPath)
        print(' * model restored ')

        return saver
