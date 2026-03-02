def main(args):
    # logger
    logger = args.logger

    # path
    input_path = args.input
    output_path = args.output

    # other configs
    val_a = args.val_a
    val_b = args.val_b
    val_c = args.val_c

    # to log smth
    logger.info('Anything you want to log, this will be logged to both console and file.')
    logger.warning('If you want to log a warning, use logger.warning(). This will also be logged to both console and file.')   
    logger.error('If you want to log an error, use logger.error(). This will also be logged to both console and file.')

    # if only log to console, use print()
    print('This will only be printed to console, not logged to file.')

    # actual code here
    result = val_a + val_b + val_c
