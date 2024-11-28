from time import sleep

from tqdm.auto import tqdm


def abbreviate_number(num):
    if num >= 1e12:
        return f"{num / 1e12:.1f}T"
    elif num >= 1e9:
        return f"{num / 1e9:.1f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.1f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.1f}k"
    else:
        return str(num)


class HumanizedTqdm(tqdm):
    @staticmethod
    def format_meter(
            n: float,
            total: float,
            elapsed: float,
            *args,
            rate:  float | None = None,
            unit: str | None = "it",
            **kwargs,
    ) -> str:
        # Abbreviate progress and total
        progress_str = abbreviate_number(n)
        total_str = abbreviate_number(total)

        if rate is None:
            rate = n / elapsed if elapsed > 0 else 0
        rate_str = abbreviate_number(rate) + f" {unit}/s" if rate else ""

        # Generate standard bar format
        bar = tqdm.format_meter(
            n, total, elapsed,
            *args,
            unit=unit,
            rate=rate,
            **kwargs
        )
        if rate_str:
            bar = bar.replace(f"{rate:.2f}{unit}/s", rate_str)

        # Replace numbers with abbreviations
        return bar.replace(f"{int(n)}/{int(total)}", f"{progress_str}/{total_str}")


if __name__ == "__main__":
    # Example usage
    total = 10**13  # 10 trillion
    bar = HumanizedTqdm(total=total, unit='tokens')
    for i in range(0, total, 10**11):  # Simulate large steps
        sleep(0.1)
        bar.update(10**11)

    bar.close()
