t_dir = "/Users/constanzafiguerola/Documents/CIS530/Project/CIS530/pos_tagged/test_pos"

f = open(t_dir, 'r')
f_out = open('only_pos_test', 'w')

f = [line.strip() for line in iter(f)]
for line in f:
	words = line.split(" ")
	for word in words:
		f_out.write(word.split("/")[1] + " ")
	f_out.write("\n")