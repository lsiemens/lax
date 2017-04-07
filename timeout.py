import signal

class timeout:
    """ 
    This class may be used as a decorator or as a context.
    
    when used as a decorator the decorated function raises a TimeoutError
    exception if the function has not returned after the maximum execution time.
    
    when used as a context the code wrapped inside the contex raises a
    TimeoutError exception if the code segment has not completed execution
    after the maximum execution time.
    
    Parameters
    ----------
    seconds : integer
        The maximum execution time for the decorated function.
    """
    def __init__(self, seconds):
        self.seconds = seconds
        self.timeout_msg = None
       
        def handle_timeout(signum, frame):
            raise TimeoutError(self.timeout_msg)
        
        signal.signal(signal.SIGALRM, handle_timeout)
        self.function = None
    
    def __call__(self, *args, **kwargs):
        if self.function is None:
            self.function = args[0]
            self.timeout_msg = "The function \"" + str(self.function.__name__) + "\" timed out after " + str(self.seconds) + " seconds."
            return self
        else:
            signal.alarm(self.seconds)
            return_value = self.function(*args, **kwargs)
            signal.alarm(0) # disable signal.alarm
            return return_value

    def __enter__(self):
        self.timeout_msg = "This context has timed out after " + str(self.seconds) + " seconds."
        signal.alarm(self.seconds)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0) # disable signal.alarm
