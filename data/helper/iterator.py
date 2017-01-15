def iterator(dataset, eval_data, distort_inputs=False, num_epochs=None,
             shuffle=False):

    def _iterate(each, done):
        # Read the inputs in batches.
        data_batch, label_batch = inputs(
            dataset, distort_inputs=distort_inputs, num_epochs=num_epochs,
            shuffle=shuffle, eval_data=eval_data)

        # Create a session to run the graph on mulitple threads.
        sess = tf.Session()
        coord = tf.train.Coordinator()

        sess.run([tf.global_variables_initializer(),
                  tf.local_variables_initializer()])

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if not eval_data and num_epochs is not None:
            num_images = dataset.num_examples_per_epoch_for_train * num_epochs
        if eval_data and num_epochs is not None:
            num_images = dataset.num_examples_per_epoch_for_eval * num_epochs

        try:
            count = 0

        while(True):
            images, labels = sess.run([image_batch, label_batch])

            for i in xrange(images.shape[0]):
                label_name = dataset.label_name(labels[i])
                image = images[i]

                # Save the image in the label named subdirectory and name it
                # incrementally.
                image_names[label_name] += 1
                image_name = '{}.png'.format(image_names[label_name])
                image_path = os.path.join(images_dir, label_name, image_name)

                imsave(image_path, image)

                count += 1

                if show_progress:
                    sys.stdout.write(
                        '\r>> Saving images to {} {:.1f}%'
                        .format(images_dir, 100.0 * count / num_images))
                    sys.stdout.flush()

    except (tf.errors.OutOfRangeError, KeyboardInterrupt):
        pass

    finally:
        coord.request_stop()
        coord.join(threads)

        sess.close()

        if show_progress:
            print('')

        print('Successfully saved {} images to {}.'.format(count, images_dir))
        pass

    return _iterate
