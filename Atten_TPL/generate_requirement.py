import pkg_resources

with open("requirements.txt", "r") as f:
    pkgs = [line.strip().split("==")[0] for line in f if line.strip()]

with open("final_requirements.txt", "w") as f_out:
    for pkg in pkgs:
        try:
            version = pkg_resources.get_distribution(pkg).version
            f_out.write(f"{pkg}=={version}\n")
        except:
            print(f"{pkg} not found in current environment")
