from config.base_config import BaseConfig

class AttackerConfig(BaseConfig):
    """Конфигурация для атакующего"""
    
    def __init__(self):
        super().__init__()
        self.strategy_type = 'offensive'
        self.aggression_factor = 0.8
        self.flank_preference = 0.6
        
        # Специфичные для атакующего награды
        self.rewards.update({
            'breakthrough': 300.0,
            'encirclement': 500.0,
            'front_advance': 20.0,
            'kill': 100.0
        })