from src.ai import human_put, ai_get_move

human_put(7,7)
ax, ay = ai_get_move()
print("AI落子坐标：", ax, ay)
