{
  lib,
  singularity-tools,
  llama-cpp,
  bashInteractive,
  interactive ? false,
}:

singularity-tools.buildImage rec {
  inherit (llama-cpp) name;
  contents = [ llama-cpp ] ++ lib.optionals interactive [ bashInteractive ];

  # These are excessive (but safe) for most variants. Building singularity
  # images requires superuser privileges, so we build them inside a VM in a
  # writable image of pre-determined size.
  #
  # Expected image sizes:
  # - cuda, all gencodes: 560M,
  diskSize = 4096;
  memSize = diskSize;
}
