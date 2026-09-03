{
  description = "The development environment for Exo";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";

    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };

    crane.url = "github:ipetkov/crane";

    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    dream2nix = {
      url = "github:nix-community/dream2nix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
    };

    # Python packaging with uv2nix
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    nixglhost = {
      url = "github:numtide/nix-gl-host";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  nixConfig = {
    extra-trusted-public-keys = "exo.cachix.org-1:okq7hl624TBeAR3kV+g39dUFSiaZgLRkLsFBCuJ2NZI= cache.nixos-cuda.org:74DUi4Ye579gUqzH4ziL9IyiJBlDpMRn9MBN8oNan9M=";
    extra-substituters = "https://exo.cachix.org https://cache.nixos-cuda.org";
  };

  outputs = inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-darwin"
        "aarch64-linux"
      ];

      imports = [
        inputs.treefmt-nix.flakeModule
        ./dashboard/parts.nix
        ./rust/parts.nix
        ./python/parts.nix
      ];

      debug = true; # Enable options autocompletion

      perSystem = { config, self', pkgs, lib, system, ... }:
        let
          pkgsArgs = {
            inherit system;
            config.allowUnfreePredicate = pkg: (pkg.pname or "") == "metal-toolchain";
            overlays = [
              inputs.nixglhost.overlays.default
              (import ./nix/apple-sdk-overlay.nix)
              (final: _: {
                macmon = final.rustPlatform.buildRustPackage {
                  pname = "macmon";
                  version = "git";
                  src = final.fetchFromGitHub {
                    owner = "vladkens";
                    repo = "macmon";
                    rev = "a1cd06b6cc0d5e61db24fd8832e74cd992097a7d";
                    hash = "sha256-wcq4PUXK44XfUKOZKl32u8LpOxXpSbUUfItQGwS2Zso=";
                  };
                  cargoHash = "sha256-Epj3L+db1flGNK5y6yfSig8piEiXTz15lPo/FNkqlkA=";
                };
              })
            ];
          };
        in
        {
          # Allow unfree for metal-toolchain (needed for Darwin Metal packages)
          _module.args = {
            pkgs = import inputs.nixpkgs pkgsArgs;
            unfreePkgs = import inputs.nixpkgs (pkgsArgs // { config.allowUnfree = true; });
          };
          treefmt = {
            projectRootFile = "flake.nix";
            # Scratch/vendored trees that NO formatter should touch. Keeping this
            # at the top level (rather than repeating it per-formatter) makes the
            # FORMAT scope match the LINT scope declared in pyproject.toml's
            # `[tool.ruff] extend-exclude`, and applies to all seven formatters at
            # once so the next throwaway probe script cannot break CI.
            #
            # This has to live here and not only in pyproject.toml: treefmt
            # invokes `ruff format` with EXPLICIT file paths, and ruff ignores
            # `exclude`/`extend-exclude` for explicitly-passed paths unless
            # `--force-exclude` is given -- so pyproject's exclusion scopes
            # `ruff check` but silently does nothing for `ruff format`. A single
            # unparseable scratch file therefore made `ruff format` exit 2 and
            # abort the entire treefmt run, so the other six formatters were
            # never reached at all.
            #
            # Patterns are matched by treefmt against the repo-root-relative path
            # using gobwas/glob compiled with no separator argument, so `**`
            # crosses `/` and there is no leading slash.
            settings.excludes = [
              ".typings/**"
              # Vendored third-party benchmark harness code -- not ours to restyle.
              "bench/vendor/**"
              # Vendored upstream dependencies, checked out as git submodules.
              "mlx/**"
              "mlx-lm/**"
              # One-off performance/diagnostic probe scripts and shell drivers.
              # Not shipped, not imported by `src/`, and already excluded from the
              # pytest run. Some are not even syntactically valid.
              "bench/**"
              "tmp/**"
            ];
            programs = {
              nixpkgs-fmt.enable = true;
              ruff-format = {
                enable = true;
                # Generated stub for the Rust extension module. Note that
                # `rust/exo_rs/**` as a whole cannot go in the global excludes
                # above, because rustfmt and taplo legitimately own the `.rs` and
                # `.toml` files in that directory.
                excludes = [ "rust/exo_rs/exo_rs.pyi" ];
              };
              rustfmt = {
                enable = true;
                package = config.rust.toolchain;
              };
              prettier = {
                enable = true;
                package = self'.packages.prettier-svelte;
                includes = [ "*.ts" "*.svelte" ];
              };
              swift-format = {
                enable = true;
                package = pkgs.swiftPackages.swift-format;
              };
              shfmt.enable = true;
              taplo.enable = true;
            };
          };

          packages = {
            default = self'.packages.exo;
          } //
          lib.optionalAttrs pkgs.stdenv.hostPlatform.isDarwin {
            metal-toolchain = pkgs.callPackage ./nix/metal-toolchain.nix { };
          };

          devShells.default = with pkgs; pkgs.mkShell {
            inputsFrom = [ self'.checks.cargo-build ];

            packages =
              [
                # FORMATTING
                config.treefmt.build.wrapper

                # PYTHON
                self'.packages.exo.passthru.evenv
                uv

                # RUST
                config.rust.toolchain
                maturin

                # NIX
                nixd
                nixpkgs-fmt

                # SVELTE
                nodejs

                # MISC
                just
                jq
              ]
              ++ lib.optionals stdenv.isDarwin [
                macmon
              ];

            OPENSSL_NO_VENDOR = "1";

            shellHook = ''
              export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${python313}/lib"
              ${lib.optionalString stdenv.isLinux ''
                export LD_LIBRARY_PATH="${openssl.out}/lib:$LD_LIBRARY_PATH"
              ''}
            '';
          };
        };
    };
}
