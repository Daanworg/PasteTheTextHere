{pkgs}: {
  deps = [
    pkgs.pkg-config
    pkgs.which
    pkgs.libpng
    pkgs.libjpeg_turbo
    pkgs.glibcLocales
  ];
}
