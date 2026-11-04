"""Prints the trainable-parameter count of every HDLM component (Table 1 and Table 6)."""
from models import build_record_cnn, BUILDERS, count_params

cnn = build_record_cnn()
cnn.summary()
print("Record CNN:", count_params(cnn))
for name, b in BUILDERS.items():
    print(name, count_params(b(14, 3, 3)))
