from SPC.UIO.ml_models.RLNNModel_Escher import RLNNModel_Escher
import numpy as np
import time
import tensorflow as tf

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
print("sess:", sess)

print(tf.config.list_physical_devices('GPU'))

iterations = 10
N = 10000
model = RLNNModel_Escher()
model.setParameters(None, 800, 20, 20)

keras = model.build_model()
x = np.random.random(size=(N, model.observation_space))

print("Input dimension:", x.shape)
t = time.time()
for i in range(iterations):
    y = keras.predict(x)
print(f"elapsed time {time.time()-t}")