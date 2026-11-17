{
  description = "Proyecto ML Grupo 9 — predicción de incumplimiento Lending Club";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in {
      devShells = forAllSystems (pkgs:
        let
          # Mismas dependencias que requirements.txt, más pytest y Jupyter.
          python = pkgs.python3.withPackages (ps: with ps; [
            pandas
            numpy
            scikit-learn
            matplotlib
            seaborn
            kaggle
            pyarrow
            fastparquet
            pytest
            ipykernel
            jupyterlab
          ]);
        in {
          default = pkgs.mkShell {
            packages = [ python pkgs.git ];
            shellHook = ''
              # Los scripts se ejecutan desde la raíz con `python -m scripts.<nombre>`
              export PYTHONPATH="$PWD:$PYTHONPATH"
              echo "Grupo 9 ML — python $(python --version | cut -d' ' -f2), sklearn $(python -c 'import sklearn;print(sklearn.__version__)')"
            '';
          };
        });
    };
}
