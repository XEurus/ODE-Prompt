"""
注册器模式实现

本模块实现了注册器模式，用于管理和查找自定义模块。
修改自 https://github.com/facebookresearch/fvcore

注册器模式的优势：
    - 解耦模块定义和使用
    - 支持通过字符串名称动态获取类
    - 便于扩展新模块

使用示例：
    # 创建注册器
    >>> BACKBONE_REGISTRY = Registry('BACKBONE')
    
    # 装饰器方式注册
    >>> @BACKBONE_REGISTRY.register()
    >>> class MyBackbone(nn.Module):
    >>>     ...
    
    # 函数调用方式注册
    >>> BACKBONE_REGISTRY.register(MyBackbone)
    
    # 获取注册的类
    >>> backbone_cls = BACKBONE_REGISTRY.get('MyBackbone')
    >>> backbone = backbone_cls()
"""

__all__ = ["Registry"]


class Registry:
    """
    注册器类
    
    提供名称到对象的映射，支持自定义模块的注册和查找。
    
    属性：
        _name: 注册器名称（用于错误消息）
        _obj_map: 名称到对象的映射字典
    
    方法：
        register: 注册对象
        get: 根据名称获取对象
        registered_names: 获取所有已注册的名称
    
    典型用法：
        1. 创建领域特定的注册器（如 BACKBONE, HEAD, TRAINER）
        2. 使用装饰器注册各种实现
        3. 通过配置文件中的字符串名称获取对应的类
    """

    def __init__(self, name):
        """
        初始化注册器
        
        参数：
            name: 注册器名称，用于错误消息
        """
        self._name = name
        self._obj_map = dict()

    def _do_register(self, name, obj, force=False):
        """
        执行注册操作
        
        参数：
            name: 对象名称
            obj: 要注册的对象（通常是类）
            force: 是否强制覆盖已存在的注册
        
        异常：
            KeyError: 名称已存在且 force=False
        """
        if name in self._obj_map and not force:
            raise KeyError(
                'An object named "{}" was already '
                'registered in "{}" registry'.format(name, self._name)
            )

        self._obj_map[name] = obj

    def register(self, obj=None, force=False):
        """
        注册对象
        
        可以作为装饰器或普通函数使用。
        
        参数：
            obj: 要注册的对象（None 表示用作装饰器）
            force: 是否强制覆盖
        
        返回：
            作为装饰器时返回包装函数，否则无返回
        
        示例：
            # 作为装饰器
            @registry.register()
            class MyClass:
                pass
            
            # 作为函数
            registry.register(MyClass)
        """
        if obj is None:
            # 用作装饰器
            def wrapper(fn_or_class):
                name = fn_or_class.__name__
                self._do_register(name, fn_or_class, force=force)
                return fn_or_class

            return wrapper

        # 用作函数调用
        name = obj.__name__
        self._do_register(name, obj, force=force)

    def get(self, name):
        """
        根据名称获取已注册的对象
        
        参数：
            name: 对象名称
        
        返回：
            已注册的对象
        
        异常：
            KeyError: 名称未注册
        """
        if name not in self._obj_map:
            raise KeyError(
                'Object name "{}" does not exist '
                'in "{}" registry'.format(name, self._name)
            )

        return self._obj_map[name]

    def registered_names(self):
        """
        获取所有已注册的名称
        
        返回：
            list: 已注册名称的列表
        """
        return list(self._obj_map.keys())
