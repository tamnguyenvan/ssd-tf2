"""
"""
import os
import time
import shutil

import yaml
from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf

from ssd import model_factory
from losses import SSDLosses
from dataset import prior_boxes
from dataset.coco import DataLoader


flags.DEFINE_string('config_file', '', 'Model config file')
flags.DEFINE_string('data_dir', 'data', 'Data directory')
flags.DEFINE_integer('batch_size', 8, 'Batch size')
flags.DEFINE_integer('epochs', 100, 'Number of epochs')
flags.DEFINE_float('lr', 1e-3, 'Learning rate')
flags.DEFINE_integer('num_classes', 81, 'Number of classes')
flags.DEFINE_integer('neg_ratio', 3, 'Negative ratio')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight decay')
flags.DEFINE_integer('num_workers', 8, 'Number of data loading threads')
flags.DEFINE_string('checkpoint_prefix', 'checkpoints/ssd',
                    'Checkpoint prefix')


@tf.function
def train_step(images, gt_confs, gt_locs, model, loss_obj, optimizer):
    with tf.GradientTape() as tape:
        confs, locs = model(images)

        conf_loss, loc_loss = loss_obj(
            confs, locs, gt_confs, gt_locs)
        loss = conf_loss + loc_loss

        l2_loss = [tf.nn.l2_loss(t) for t in model.trainable_variables]
        l2_loss = FLAGS.weight_decay * tf.math.reduce_sum(l2_loss)
        loss += l2_loss

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, conf_loss, loc_loss, l2_loss


@tf.function
def test_step(images, gt_confs, gt_locs, model, loss_obj):
    confs, locs = model(images)

    conf_loss, loc_loss = loss_obj(
        confs, locs, gt_confs, gt_locs)
    loss = conf_loss + loc_loss

    l2_loss = [tf.nn.l2_loss(t) for t in model.trainable_variables]
    l2_loss = FLAGS.weight_decay * tf.math.reduce_sum(l2_loss)
    loss += l2_loss

    return loss, conf_loss, loc_loss, l2_loss


def main(_argv):
    devices = tf.config.experimental.list_physical_devices('GPU')
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)

    # Uncomment for debugging only
    # tf.config.run_functions_eagerly(True)
    # tf.debugging.enable_check_numerics()

    # Load config
    if not os.path.exists(FLAGS.config_file):
        logging.error('Not found config file')
        return
    with open(FLAGS.config_file) as f:
        cfg = yaml.load(f, yaml.FullLoader)

    # Load data
    logging.info('Loading train data')
    priors = prior_boxes.PriorBox(cfg).forward()
    anno_dir = os.path.join(FLAGS.data_dir, 'annotations')
    label_map_file = os.path.join(FLAGS.data_dir, 'coco.names')

    train_image_dir = os.path.join(FLAGS.data_dir, 'train2017')
    train_anno_path = os.path.join(anno_dir, 'instances_train2017.json')
    train_loader = DataLoader(priors, batch_size=FLAGS.batch_size,
                              num_workers=FLAGS.num_workers,
                              image_size=cfg['INPUT']['IMAGE_SIZE'],
                              training=True)
    train_data, num_train = train_loader.load(
        train_image_dir, label_map_file, train_anno_path)

    logging.info('Loading val data')
    val_image_dir = os.path.join(FLAGS.data_dir, 'val2017')
    val_anno_path = os.path.join(anno_dir, 'instances_val2017.json')
    val_loader = DataLoader(priors, batch_size=FLAGS.batch_size,
                            num_workers=FLAGS.num_workers,
                            image_size=cfg['INPUT']['IMAGE_SIZE'],
                            training=False)
    val_data = val_loader.load(val_image_dir, label_map_file, val_anno_path)

    # Create the model + optimizer
    model = model_factory.create_model(cfg)
    model_name = cfg['MODEL']['NAME']
    logging.info(f'Created model {model_name}')
    # model.summary()
    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.lr)

    # Create a checkpoint for smooth training
    ckpt_prefix = os.path.join(FLAGS.checkpoint_prefix, model_name.lower())
    ckpt = tf.train.Checkpoint(
        step=tf.Variable(0), optimizer=optimizer,
        model=model, train_data=train_data)
    manager = tf.train.CheckpointManager(ckpt, ckpt_prefix, max_to_keep=1)

    # Retore variables if checkpoint exists
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        logging.info('Restoring from {}'.format(manager.latest_checkpoint))
    else:
        logging.info('Train the model from scratch')

    # Criterion
    loss_obj = SSDLosses(FLAGS.neg_ratio, FLAGS.num_classes)

    # Loss aggregation
    train_loss = tf.keras.metrics.Mean(name='loss')
    train_conf_loss = tf.keras.metrics.Mean(name='conf_loss')
    train_loc_loss = tf.keras.metrics.Mean(name='loc_loss')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_conf_loss = tf.keras.metrics.Mean(name='val_conf_loss')
    val_loc_loss = tf.keras.metrics.Mean(name='val_loc_loss')

    # Tensorboard
    if os.path.exists('logs'):
        shutil.rmtree('logs')

    train_log_dir = 'logs/train'
    train_writer = tf.summary.create_file_writer(train_log_dir)

    val_log_dir = 'logs/val'
    val_writer = tf.summary.create_file_writer(val_log_dir)
    steps_per_epoch = num_train // FLAGS.batch_size
    for epoch in range(FLAGS.epochs):
        start = time.time()
        for batch, (images, gt_confs, gt_locs) in enumerate(train_data):
            loss, conf_loss, loc_loss, l2_loss = train_step(
                images, gt_confs, gt_locs, model, loss_obj, optimizer)

            ckpt.step.assign_add(1)
            if int(ckpt.step) % 500 == 0:
                save_path = manager.save()
                logging.info('Saved checkpoint for step {}: {}'.format(
                    int(ckpt.step), save_path))

            if (batch + 1) % 10 == 0:
                logging.info('Epoch {:03d} iter {:06d}/{:06d} | '
                             'total_loss: {:.2f} conf_loss: {:.2f} '
                             'loc_loss: {:.2f}'.format(
                                 epoch + 1, int(ckpt.step),
                                 steps_per_epoch, loss.numpy(),
                                 conf_loss.numpy(), loc_loss.numpy()))

            train_loss.update_state(loss)
            train_conf_loss.update_state(conf_loss)
            train_loc_loss.update_state(loc_loss)

        with train_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch+1)
            tf.summary.scalar('conf_loss', train_conf_loss.result(), step=epoch+1)
            tf.summary.scalar('loc_loss', train_loc_loss.result(), step=epoch+1)

        for batch, (images, gt_confs, gt_locs) in enumerate(val_data):
            loss, conf_loss, loc_loss, l2_loss = test_step(
                images, gt_confs, gt_locs, model, loss_obj
            )
            val_loss.update_state(loss)
            val_conf_loss.update_state(conf_loss)
            val_loc_loss.update_state(loc_loss)
        logging.info('Evaluation | Epoch {:03d} | total_loss: {:.2f} '
                     'conf_loss: {:.2f} loc_loss: {:.2f}'.format(
                         epoch + 1, val_loss.result(), val_conf_loss.result(),
                         val_loc_loss.result()))

        with val_writer.as_default():
            tf.summary.scalar('loss', val_loss.result(), step=epoch+1)
            tf.summary.scalar('conf_loss', val_conf_loss.result(), step=epoch+1)
            tf.summary.scalar('loc_loss', val_loc_loss.result(), step=epoch+1)

        train_loss.reset_states()
        train_conf_loss.reset_states()
        train_loc_loss.reset_states()

        val_loss.reset_states()
        val_conf_loss.reset_states()
        val_loc_loss.reset_states()

        print('Epoch {} took {:.2f}'.format(epoch + 1, time.time() - start))

    model.save_weights(ckpt_prefix)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
