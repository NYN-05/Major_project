from pathlib import Path

p = Path(__file__).resolve().parent / 'test_video.avi'
print('EXISTS', p.exists())
print('SIZE', p.stat().st_size if p.exists() else None)
