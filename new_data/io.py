def write(num_examples_per_epoch, input_filenames, output_filename, read,
          preprocess=None, epochs=EPOCHS, batch_size=BATCH_SIZE,
          eval_data=False, dataset_name='', show_progress=True):

    steps = -(-num_examples_per_epoch // batch_size) * epochs

    filename_queue = tf.train.string_input_producer(input_filenames)

    record = read(filename_queue)

    record = preprocess(record)

    data_batch, label_batch = tf.train.batch(
        [record.data, record.label],
        batch_size=batch_size,
        num_threads=NUM_THREADS,
        capacity=CAPACITY)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    writer = tf.python_io.TFRecordWriter(output_filename)

    start_time = time.time()

    try:
        for i in xrange(1, steps+1):
            data, labels = sess.run([data_batch, label_batch])

            for j in xrange(batch_size):
                example = get_example(data[j], int(labels[j][0]))
                writer.write(example.SerializeToString())

            remaining = (steps - i) * ((time.time() - start_time) / i) / 60

            if show_progress:
                sys.stdout.write(' '.join([
                    '\r>> Writing {}'.format(dataset_name),
                    '{}'.format('eval' if eval_data else 'train'),
                    'dataset to',
                    '{}'.format(output_filename),
                    '{:.1f}%'.format(100*i/steps),
                    '-',
                    '{:.1f} min remaining'.format(remaining),
                ]))
                sys.stdout.flush()

    except KeyboardInterrupt:
        pass

    finally:
        coord.request_stop()
        coord.join(threads)

        writer.close()
        sess.close()

        if show_progress:
            print('')

        print(' '.join([
            'Successfully written {} example'.format(i * batch_size),
            '({:.2f} epochs).'.format(i * batch_size / num_examples_per_epoch),
        ]))
