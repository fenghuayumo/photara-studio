# Averages the SPLAT_IGS_PROFILE refine lines of one or more runs, so two
# builds can be compared without depending on a single (noisy) refine event.
param([Parameter(Mandatory = $true)][string[]]$Log)

foreach ($path in $Log) {
    $lines = Select-String -Path $path -Pattern 'igs_profile' |
        ForEach-Object { $_.Line }
    if ($lines.Count -eq 0) {
        Write-Output ("{0}: no igs_profile lines" -f $path)
        continue
    }
    $total = @(); $wait = @(); $hits = @(); $miss = @(); $missMib = @(); $drop = @()
    foreach ($line in $lines) {
        $sum = 0.0
        foreach ($m in [regex]::Matches($line, '([a-z_0-9]+)_ms=([0-9.]+)')) {
            $sum += [double]$m.Groups[2].Value
        }
        $total += $sum
        $wait += [double]([regex]::Match($line, '(?<![a-z_])wait_ms=([0-9.]+)').Groups[1].Value)
        $hits += [double]([regex]::Match($line, 'pool_hit=(\d+)').Groups[1].Value)
        $miss += [double]([regex]::Match($line, 'pool_miss=(\d+)').Groups[1].Value)
        $missMib += [double]([regex]::Match($line, 'pool_miss_mib=([0-9.]+)').Groups[1].Value)
        $drop += [double]([regex]::Match($line, 'pool_drop_mib=([0-9.]+)').Groups[1].Value)
    }
    $avg = { param($values) ($values | Measure-Object -Average).Average }
    Write-Output ("{0}: n={1} refine_ms={2:N1} wait_ms={3:N1} pool_hit={4:N0} pool_miss={5:N0} miss_MiB={6:N0} drop_MiB={7:N0}" -f `
        (Split-Path -Leaf $path), $lines.Count, (& $avg $total), (& $avg $wait), `
        (& $avg $hits), (& $avg $miss), (& $avg $missMib), (& $avg $drop))
}
