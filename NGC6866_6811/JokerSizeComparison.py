import os
import thejoker as tj

path_200 = "/data/labs/douglste-laf-lab/wichmand/stardata/200.0M_new"
path_200_jitter = "/data/labs/douglste-laf-lab/wichmand/stardata/200.0M_jitter"
path_400_jitter = "/data/labs/douglste-laf-lab/wichmand/stardata/400.0M_jitter"


def get_num_samples(filename):
    try:
        return len(tj.JokerSamples.read(filename))
    except Exception:
        return None


print(f"{'Star ID':<20} {'200M':>8} {'200M_jitter':>14} {'400M_jitter':>14}")
print("-" * 60)

for star in sorted(os.listdir(path_400_jitter)):
    star_dir = os.path.join(path_400_jitter, star)
    if not os.path.isdir(star_dir):
        continue

    file_400 = os.path.join(
        star_dir,
        f"rejection_samples_400M_jitter_{star}.hdf5"
    )

    file_200j = os.path.join(
        path_200_jitter,
        star,
        f"rejection_samples_200M_jitter_{star}.hdf5"
    )

    file_200 = os.path.join(
        path_200,
        star,
        f"rejection_samples_200.0M_{star}_new.hdf5"
    )

    n200 = get_num_samples(file_200)
    n200j = get_num_samples(file_200j)
    n400j = get_num_samples(file_400)

    print(f"{star:<20} {str(n200):>8} {str(n200j):>14} {str(n400j):>14}")
