import argparse
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from losses.cloth import Cloth
from losses.material import Material
from losses.physics import gravitational_energy,stretching_energy,stretching_energy2,bending_energy,collision_penalty,inertial_term_sequence
import smpl
from smpl import LBS
import snug_utils as utils
from losses.utils import fix_collisions,fix_collisions_sequence
from snug_utils import loadInfo,weights_prior, my_weights_prior
from dataset_amass import AMASS




# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class CustomCallback(tf.keras.callbacks.Callback):
   def __init__(self, gru_layer, w):
        self.gru_layer = gru_layer
        self.w = w
   def on_epoch_end(self, epoch, logs=None):
        self.gru_layer.reset_states(self.w)


class SnugModel(tf.keras.Model):

    def __init__(self,cloth):
        super(SnugModel,self).__init__()

        # build
        self.cloth = cloth
        self.v_TGRAMENT = cloth.v_template
        self._build()
        self.body = smpl.SMPL("assets/SMPL/basicModel_f_lbs_10_207_0_v1.0.0.pkl")
        # outfit data
        self._T = cloth.v_template
        self.window = 3


        cloth.compute_skinning_weights(self.body)
        self._W = cloth.v_weights
        self.g_skinweight = cloth.v_weights



        print('SNUG model initialized.')


    # def call(self,poses,trans_vel,betas,G, translation,shapeblend, window=3,training=False):
    def call(self, shape, pose, translation, trans_vel, hidden_states0,hidden_states1,hidden_states2,hidden_states3):
        #ipnput 16x3x85
        # G: 48x24x4x4
        # translation 48x3

        num_vertices = self.cloth.num_vertices
        #x : Bsize X 3 x 82
        x = tf.concat([shape, pose], axis=-1)
        x = self.gru_layer1(x, initial_state=hidden_states0)
        x = self.gru_layer2(x, initial_state=hidden_states1)
        x = self.gru_layer3(x, initial_state=hidden_states2)
        x = self.gru_layer4(x, initial_state=hidden_states3)
        x = self.Linear(x)


        x = tf.reshape(x,[-1,3,num_vertices,3])
        x0 = x[:,0,:]
        x1 = x[:,1,:]
        x2 = x[:,2,:]

        pose0 = pose[:,0,:]
        pose1 = pose[:,1,:]
        pose2 = pose[:,2,:]
        translation0 = translation[:,0,:]
        translation1 = translation[:,1,:]
        translation2 = translation[:,2,:]


        v_body0, tensor_dict0 = body(
            shape=tf.reshape(shape[:,0,:], [-1,1, 10]),
            pose=tf.reshape(pose0, [-1, 1, 72]),
            translation=tf.reshape(translation0, [-1, 3]),
        )
        #v_garment_offset0 = x0+tf.reshape(self.v_TGRAMENT,[-1,3])
        v_garment_offset0 = x0 + self.cloth.v_template
        v_garment_skinning0 = smpl.LBS()(v_garment_offset0, tensor_dict0["joint_transforms"], self.g_skinweight)
        # v_garment_skinning0 = self._skinning(v_garment_offset0, self.g_skinweight)

        v_garment_skinning0 += translation0[:, tf.newaxis, :]
        v_garment_skinning0 = tf.reshape(v_garment_skinning0,[-1,num_vertices,3])

        v_body1, tensor_dict1 = body(
            shape=tf.reshape(shape[:,1,:], [-1, 1, 10]),
            pose=tf.reshape(pose1, [-1, 1, 72]),
            translation=tf.reshape(translation1, [-1, 3]),
        )

        v_garment_offset1 = x1 + self.cloth.v_template
        v_garment_skinning1 = smpl.LBS()(v_garment_offset1, tensor_dict1["joint_transforms"], self.g_skinweight)
        v_garment_skinning1 += translation1[:, tf.newaxis, :]
        v_garment_skinning1 = tf.reshape(v_garment_skinning1, [-1, num_vertices, 3])

        v_body2, tensor_dict2 = body(
            shape=tf.reshape(shape[:,2,:], [-1, 1, 10]),
            pose=tf.reshape(pose2, [-1, 1, 72]),
            translation=tf.reshape(translation2, [-1, 3]),
        )

        v_garment_offset2 = x2 + self.cloth.v_template
        v_garment_skinning2 = smpl.LBS()(v_garment_offset2, tensor_dict2["joint_transforms"], self.g_skinweight)
        v_garment_skinning2 += translation2[:, tf.newaxis, :]
        v_garment_skinning2 = tf.reshape(v_garment_skinning2, [-1, num_vertices, 3])



        return v_garment_skinning0, v_garment_skinning1, v_garment_skinning2,v_body2,tensor_dict2

        # disp = tf.reshape(x4__, [self.window, batch_size, 4424, 3])
        # disp_ = tf.transpose(disp, perm=[1, 0, 2, 3]) # NT.. -> TN..
        #
        # v_p = disp_+ self.cloth.v_template #v_garment # sifirla carptim sadece skinning calisiyorum su an
        # # v_p = v_p + tf.reshape(translation, (batch_size, 3, 1, 3))
        # # Compute skinning
        # v_f = self._skinning(v_p, G)
        # v_f = v_f + tf.reshape(translation, (batch_size, self.window, 1, 3))
        # return v_f

    def _run(self, shape, pose, translation, trans_vel, hidden_states0,hidden_states1,hidden_states2,hidden_states3):
        #ipnput 16x3x85
        # G: 48x24x4x4
        # translation 48x3
        num_vertices = self.cloth.num_vertices

        x = tf.concat([shape, pose], axis=-1)
        x = self.gru_layer1(x, initial_state=hidden_states0)
        x = self.gru_layer2(x, initial_state=hidden_states1)
        x = self.gru_layer3(x, initial_state=hidden_states2)
        x = self.gru_layer4(x, initial_state=hidden_states3)
        x = self.Linear(x)

        x = tf.reshape(x,[-1,1,4424,3])
        x0 = x[:,0,:]
        # x1 = x[:,1,:]
        # x2 = x[:,2,:]

        pose0 = pose[:,0,:]
        # pose1 = pose[:,1,:]
        # pose2 = pose[:,2,:]
        translation0 = translation[:,0,:]
        # translation1 = translation[:,1,:]
        # translation2 = translation[:,2,:]


        v_body0, tensor_dict0 = body(
            shape=tf.reshape(shape[:,0,:], [-1,1, 10]),
            pose=tf.reshape(pose0, [-1, 1, 72]),
            translation=tf.reshape(translation0, [-1, 3]),
        )
        v_garment_offset0 = x0+tf.reshape(self.v_TGRAMENT,[-1,3])
        v_garment_skinning0 = smpl.LBS()(v_garment_offset0, tensor_dict0["joint_transforms"], self.g_skinweight)
        v_garment_skinning0 += translation0[:, tf.newaxis, :]
        v_garment_skinning0 = tf.reshape(v_garment_skinning0,[-1,num_vertices,3])

        return v_garment_skinning0,v_body0,tensor_dict0

    # Builds model
    def _build(self):


        # self.g_skinweight = garment_skinning_weights
        self.gru_layer1 = tf.keras.layers.GRU(256, dropout=0.1, recurrent_dropout=0.2, return_sequences=True, use_bias=True, activation='tanh')
        self.gru_layer2 = tf.keras.layers.GRU(256, dropout=0.1, recurrent_dropout=0.2, return_sequences=True, use_bias=True, activation='tanh')
        self.gru_layer3 = tf.keras.layers.GRU(256, dropout=0.1, recurrent_dropout=0.2, return_sequences=True, use_bias=True, activation='tanh')
        self.gru_layer4 = tf.keras.layers.GRU(256, dropout=0.1, recurrent_dropout=0.2, return_sequences=True, use_bias=True, activation='tanh')

        #self._W = tf.Variable(self._W, name='blendweights', trainable=self._blendweights)

        self.Linear = tf.keras.layers.Dense(units=4424 * 3)


    # Computes the skinning for each outfit/pose
    def _skinning(self, T, G):

        b,t = tf.shape(T)[0:2]

        T=tf.reshape(T,(b*t,4424,3))

        total = LBS()(T, G, self._W)

        total_=tf.reshape(total, (b, t, 4424, 3))

        return total_

    def _with_ones(self, X):
        return tf.concat((X, tf.ones((*X.shape[:2], 1), tf.float32)), axis=-1)


if __name__ == "__main__":
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--motion",
        type=str,
        default="assets/CMU/07/07_02_poses.npz",
        help="path of the motion to use as input"
    )

    parser.add_argument(
        "--garment",
        type=str,
        default="tshirt",
        help="name of the garment (tshirt, tank, top, pants or shorts)"
    )

    parser.add_argument(
        "--savedir",
        type=str,
        default="tmp",
        help="path to save the result"
    )

    args = parser.parse_args()
    
    # Load smpl
    body = smpl.SMPL("assets/SMPL/basicModel_f_lbs_10_207_0_v1.0.0.pkl")

    # Load garment model
    model_path, template_path = utils.get_model_path(args.garment)
    # snug = tf.saved_model.load(model_path)
    v_garment, f_garment = utils.load_obj(template_path)

    # # Body shape
    # betas = np.zeros(10, dtype=np.float32)


    # Fabric material parameters
    thickness = 0.00047 # (m)
    bulk_density = 426  # (kg / m3)
    area_density = thickness * bulk_density

    material = Material(
        density=area_density, # Fabric density (kg / m2)
        thickness=thickness,  # Fabric thickness (m)
        young_modulus=0.7e5,
        poisson_ratio=0.485,
        stretch_multiplier=1,
        bending_multiplier=50
    )

    print(f"Lame mu {material.lame_mu:.2E}, Lame lambda: {material.lame_lambda:.2E}")

    # Initialize structs
    cloth = Cloth(
        path="assets/meshes/tshirt.obj",
        material=material,
        dtype=tf.float32
    )

    snug = SnugModel(cloth)

    # Instantiate an optimizer.
    # optimizer = keras.optimizers.SGD(learning_rate=0.001)
    # optimizer = keras.optimizers.Adam(clipnorm=1.0)
    optimizer = keras.optimizers.Adam()
    #https://cnvrg.io/gradient-clipping/
    print(optimizer)
    epochs = 10
    batch_size = 4# 16
    body = smpl.SMPL("assets/SMPL/basicModel_f_lbs_10_207_0_v1.0.0.pkl")

    mydata = AMASS('assets/CMU2')



    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        iterator = iter(mydata.dataset.batch(batch_size).shuffle(seed=42,buffer_size=500))
        # Iterate over the batches of the dataset.

        next_exist = True
        step = 0
        while next_exist:
            optional = iterator.get_next_as_optional()
            if not optional.has_value():
                next_exist = False
                break
                ################################################################

            poses, trans, trans_vel,g = optional.get_value()

            bs_size, window = poses.shape[0], poses.shape[1]

            # betas = np.zeros(bs_size*window*10, dtype=np.float32)
            # shape = tf.reshape(betas, (bs_size, window, 10))
            # shape = tf.zeros([bs_size, window, 10], dtype=tf.float32) ## redundant
            shape = tf.random.uniform(shape=[bs_size, window, 10], maxval=3,minval=-3)
            betas = shape*1.0

            hidden_states = [
                tf.random.normal([bs_size, 256], mean=0, stddev=0.1, dtype=tf.float32),  # State 0
                tf.random.normal([bs_size, 256], mean=0, stddev=0.1, dtype=tf.float32),  # State 1
                tf.random.normal([bs_size, 256], mean=0, stddev=0.1, dtype=tf.float32),  # State 2
                tf.random.normal([bs_size, 256], mean=0, stddev=0.1, dtype=tf.float32),  # State 3
            ]

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                ###########################
                v_body, tensor_dict = body(
                    shape=tf.reshape(betas, [-1, 10]),
                    pose=tf.reshape(poses, [-1, 72]),
                    translation=tf.reshape(trans, [-1, 3]),
                )

                g_joint_transforms = tensor_dict['joint_transforms']
                shape_blends = tensor_dict['shape_blendshape']

                translation = tf.reshape(trans, [-1, 3])

                pred1, pred2, pred3, v_body0, body_tensor_dict = snug(
                    shape,
                    poses,
                    trans,
                    trans_vel,
                    hidden_states[0][:bs_size,:],
                    hidden_states[1][:bs_size,:],
                    hidden_states[2][:bs_size,:],
                    hidden_states[3][:bs_size,:] )


                # pred1 = tf.expand_dims(pred1, 1)
                # pred2 = tf.expand_dims(pred2, 1)
                # pred3 = tf.expand_dims(pred3, 1)
                # logits = tf.concat([pred1, pred2, pred3], axis=1)
                # logits = tf.reshape(logits, shape=(bs_size, window, 4424, 3))

                logits = tf.concat([pred1, pred2, pred3], axis=1)
                logits = tf.reshape(logits, shape=(bs_size, window, 4424, 3))
                # v_body = tf.reshape(v_body,(bs_size*window,6890,3))

                inertia = inertial_term_sequence(
                    x=logits,
                    mass=snug.cloth.v_mass,
                    time_step=1 / 30
                )


                logits = tf.reshape(logits, (-1, 4424, 3))

                # Compute the loss value for this minibatch.

                energy_stretch = stretching_energy2(v=logits,cloth=snug.cloth)  #*0.01#nan donderiyor
                energy_bend = bending_energy(logits,cloth=snug.cloth)
                g_energy = gravitational_energy(logits, mass=snug.cloth.v_mass)
                penalty = collision_penalty(logits, v_body, tensor_dict["vertex_normals"])

                loss_value = 1 * energy_bend + 1 * g_energy + 1 * inertia + 1*energy_stretch + penalty * 1


            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, snug.trainable_weights)

            # grads = [(tf.clip_by_value(grad, clip_value_min=-10.0, clip_value_max=10.0)) for grad in grads]

            # Run one step of gradient descent by updating the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, snug.trainable_weights))


            # Log every 200 batches.
            if step % 50 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                penalty = collision_penalty(logits, v_body, tensor_dict["vertex_normals"])
                print(f"Bending Energy:{float(energy_bend)},Gravitation: {float(g_energy)},Inertia:{float(inertia)}, Stretch: {energy_stretch},Penalty: {penalty}")
                # print(
                #     f"Bending Energy:{float(energy_bend)},Gravitation: {float(g_energy)},Inertia:{float(inertia)}, Penalty: {penalty}")
                print(epoch,"Seen so far: %s samples" % ((step + 1) * batch_size))

            if step % 250 == 0:  # epoch == 0:
                print('son batchimi aldim', step)
                #frame = 0  # bu kisim tek bir tane frame icin input verecek.
                garment_path = os.path.join(args.savedir, f"hus_{epoch}_step_{step:04d}_{args.garment}.obj")
                utils.save_obj(garment_path, logits[0], f_garment)
                if 0:
                    v_garment = fix_collisions(
                        vc=logits[0],
                        vb=v_body[0],
                        nb=tensor_dict["vertex_normals"][0]  # body vertex normals
                        # v_indices=v_indices  # nearest neighbour indices
                    )
                    garment_path = os.path.join(args.savedir, f"hus_{epoch}__fixed_step_{step:04d}_{args.garment}.obj")
                    utils.save_obj(garment_path, v_garment, f_garment)

            step=step+1
    ############################################################################################


    snug.save_weights('trained_models/adam_ep10v2/ep10-adam', overwrite=True, save_format=None, options=None)

    restored = SnugModel(cloth)

    restored.load_weights('trained_models/adam_ep10v2/ep10-adam', by_name=False, skip_mismatch=False, options=None)

    ############################################################################################

    print('Training is done.')
    print('Begin evaluation.')

    # Load motion
    poses, trans, trans_vel = utils.load_motion(args.motion)
    # Body shape
    betas = np.zeros(10, dtype=np.float32)

    batch_size = 1
    hidden_states = [
        tf.zeros((batch_size , 256), dtype=tf.float32),  # State 0
        tf.zeros((batch_size , 256), dtype=tf.float32),  # State 1
        tf.zeros((batch_size , 256), dtype=tf.float32),  # State 2
        tf.zeros((batch_size , 256), dtype=tf.float32),  # State 3
    ]

    for frame in range(len(poses)):
        pose = tf.reshape(poses[frame], [1, 1, 72])
        shape = tf.reshape(betas, (1, 1, 10))
        translation = tf.reshape(trans[frame], (1, 1, 3))
        translation_vel = tf.reshape(trans_vel[frame], (1, 1, 3))

        # Eval body
        v_body, tensor_dict = body(
            shape=tf.reshape(shape, [-1, 10]),
            pose=tf.reshape(pose, [-1, 72]),
            translation=tf.reshape(translation, [-1, 3]),
        )


        g_joint_transforms = tensor_dict['joint_transforms']
        shape_blends = tensor_dict['shape_blendshape']


        v_garment, v_body0, body_tensor_dict = restored._run(
            shape, pose, translation, translation_vel, hidden_states[0], hidden_states[1], \
            hidden_states[2], hidden_states[3])


        v_garment = tf.reshape(v_garment, (-1, 4424, 3))
        v_garment = fix_collisions_sequence(
            vc=v_garment[0][None],
            vb=v_body0,
            nb=tensor_dict["vertex_normals"]  # body vertex normals
            # v_indices=v_indices  # nearest neighbour indices
        )

        body_path = os.path.join(args.savedir, f"{frame:04d}_body.obj")
        utils.save_obj(body_path, v_body0, body.faces)

        garment_path = os.path.join(args.savedir, f"{frame:04d}_{args.garment}.obj")
        utils.save_obj(garment_path, v_garment, f_garment)

