from config.base_config import BaseConfig

class DefenderConfig(BaseConfig):
    """Конфигурация для защитника"""
    
    def __init__(self):
        super().__init__()
        self.strategy_type = 'defensive'
        self.defense_depth = 0.7
        self.reserve_preference = 0.4
        
        # Специфичные для защитника награды
        self.rewards.update({
            'hold_position': 100.0,
            'counter_attack': 200.0,
            'attrition': 50.0,
            'kill': 80.0
        })