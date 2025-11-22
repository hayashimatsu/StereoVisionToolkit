import functools
import inspect
import json
import traceback

def error_handler(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # 獲取當前幀的信息
            current_frame = inspect.currentframe()
            # 獲取調用者的幀
            caller_frame = current_frame.f_back
            # 獲取錯誤發生的文件名、行號和函數名
            file_name = caller_frame.f_code.co_filename
            line_number = caller_frame.f_lineno
            function_name = caller_frame.f_code.co_name
            # 獲取類名（如果在類方法中）
            class_name = args[0].__class__.__name__ if args else "Not in a class"
            
            error_message = (
                f"Error in {class_name}.{function_name} "
                f"at {file_name}:{line_number}\n"
                f"Accessing attribute: {args[1] if len(args) > 1 else 'Unknown'}\n"
                f"Error: {str(e)}"
            )
            # print(error_message)
            # 如果你想要程序繼續運行，可以返回一個默認值或者None
            # return None
            # 如果你想要程序在遇到錯誤時停止，可以取消下面這行的注釋
            raise

    return wrapper