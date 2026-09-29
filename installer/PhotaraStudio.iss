; Compile through scripts\package_windows.ps1. Command-line defines make this
; file reusable for every version and keep the installer separate from the build.
#ifndef MyAppVersion
  #define MyAppVersion "0.0.0-dev"
#endif
#ifndef MySourceDir
  #error "MySourceDir must point to the prepared portable package directory."
#endif
#ifndef MyOutputDir
  #define MyOutputDir "Output"
#endif
#ifndef MySetupIconFile
  #error "MySetupIconFile must point to apps\icons\icon.ico."
#endif

[Setup]
AppId={{C4617FCD-4F70-40B8-A4FB-0753E21C69DD}
AppName=Photara Studio
AppVersion={#MyAppVersion}
VersionInfoVersion={#MyAppVersion}
AppPublisher=Photara
DefaultDirName={autopf}\Photara Studio
DefaultGroupName=Photara Studio
DisableProgramGroupPage=yes
LicenseFile={#MySourceDir}\LICENSE
SetupIconFile={#MySetupIconFile}
CloseApplications=yes
OutputDir={#MyOutputDir}
OutputBaseFilename=Photara-Studio-{#MyAppVersion}-win64-setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
PrivilegesRequiredOverridesAllowed=dialog commandline
UninstallDisplayIcon={app}\photara_studio.exe

[Files]
Source: "{#MySourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Photara Studio"; Filename: "{app}\photara_studio.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\Photara Studio"; Filename: "{app}\photara_studio.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Run]
Filename: "{app}\photara_studio.exe"; Description: "Launch Photara Studio"; Flags: nowait postinstall skipifsilent
