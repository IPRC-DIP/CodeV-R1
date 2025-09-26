import datetime
from itertools import combinations, product
import json
import math
import os
import random
import re
import shutil
import subprocess
import networkx as nx


def extract_verilog_code(file_content):
    note_pattern = r"(//[^\n]*|/\*[\s\S]*?\*/)"
    file_content = re.sub(note_pattern, "", file_content)
    file_content = re.sub(r"(?:\s*?\n)+", "\n", file_content)

    define_pattern = r"`define\b\s+\b([a-zA-Z_][a-zA-Z0-9_$]*|\\[!-~]+?(?:\s|$))\b.*\n"

    module_pattern = r"\bmodule\s+([a-zA-Z_][a-zA-Z0-9_$]*|\\[!-~]+?(?:\s|$))\s*(?:\#\s*\([\s\S]*?\)\s*)?\((?:(?!\bmodule\b).)*?\)\s*;(?:(?!\bmodule\b).)*?\bendmodule\b"

    item_dict = {}
    item_order = []
    for match in re.finditer(
        f"{module_pattern}|{define_pattern}", file_content, re.DOTALL
    ):
        item_name = match.group(1)
        if item_name not in item_order:
            item_order.append(item_name)
        item_dict[item_name] = match.group(0)

    extracted = "\n".join([item_dict[item] for item in item_order])

    return extracted


class eda_tools:

    def __init__(
        self,
        golden_suffix="_gold",
        gate_suffix="_gate",
        use_directed_tests=False,
        random_seq_steps=1000,
        random_seq_num=100,
        quiet=False,
    ):
        self.golden_suffix = golden_suffix
        self.gate_suffix = gate_suffix
        self.random_seq_steps = random_seq_steps
        self.random_seq_num = random_seq_num
        self.quiet = quiet

    def auto_top(self, verilog_code):
        instance_graph = nx.DiGraph()
        note_pattern = r"(//[^\n]*|/\*[\s\S]*?\*/)"
        new_code = re.sub(note_pattern, "", verilog_code)
        new_code = re.sub(r"(?:\s*?\n)+", "\n", new_code)
        module_def_pattern = r"(module\s+)([a-zA-Z_][a-zA-Z0-9_\$]*|\\[!-~]+?(?=\s))(\s*\#\s*\([\s\S]*?\))?(\s*(?:\([^;]*\))?\s*;)([\s\S]*?)?(endmodule)"
        module_defs = re.findall(module_def_pattern, new_code, re.DOTALL)
        if not module_defs:
            raise Exception("No module found in auto_top().")
        module_names = [m[1] for m in module_defs]
        instance_graph.add_nodes_from(module_names)
        for mod in module_defs:
            this_mod_name = mod[1]
            this_mod_body = mod[4]
            for submod in module_names:
                if submod != this_mod_name:
                    module_instance_pattern = rf"({re.escape(submod)})(\s)(\s*\#\s*\([\s\S]*?\))?([a-zA-Z_][a-zA-Z0-9_\$]*|\\[!-~]+?(?=\s))(\s*(?:\([^;]*\))?\s*;)"
                    module_instances = re.findall(
                        module_instance_pattern, this_mod_body, re.DOTALL
                    )
                    if module_instances:
                        instance_graph.add_edge(this_mod_name, submod)
        instance_tree_size = {}
        for n in instance_graph.nodes:
            if instance_graph.in_degree(n) == 0:
                instance_tree_size[n] = nx.descendants(instance_graph, n)
        top_module = max(instance_tree_size, key=instance_tree_size.get)
        return top_module

    def process_verilog(self, verilog_code, suffix):
        note_pattern = r"(//[^\n]*|/\*[\s\S]*?\*/)"
        new_code = re.sub(note_pattern, "", verilog_code)
        new_code = re.sub(r"(?:\s*?\n)+", "\n", new_code)
        module_def_pattern = r"(module\s+)([a-zA-Z_][a-zA-Z0-9_\$]*|\\[!-~]+?(?=\s))(\s*\#\s*\([\s\S]*?\))?(\s*(?:\([^;]*\))?\s*;)([\s\S]*?)?(endmodule)"
        module_defs = re.findall(module_def_pattern, new_code, re.DOTALL)
        module_names = [m[1] for m in module_defs]
        for submod in module_names:
            module_instance_pattern = rf"({submod})(\s+)(\#\s*\([\s\S]*?\)\s*)?([a-zA-Z_][a-zA-Z0-9_\$]*|\\[!-~]+?(?=\s))(\s*(?:\([^;]*\))?\s*;)"
            new_code = re.sub(module_instance_pattern, rf"\1{suffix}\2\3\4\5", new_code)
        new_code = re.sub(module_def_pattern, rf"\1\2{suffix}\3\4\5\6", new_code)
        return new_code

    def generate_testbench(
        self,
        input_port_width,
        output_port_width,
        clock_port_polarity,
        reset_port_polarity_sync,
        golden_top,
        gate_top,
    ):
        reset_port_names = set([p[0] for p in reset_port_polarity_sync])
        if len(clock_port_polarity) > 1:
            raise Exception(
                "Multiple clock ports or multiple triggering edge detected, currently not supported."
            )

        clock_port_name = (
            list(clock_port_polarity)[0][0] if clock_port_polarity else None
        )
        clock_port_edge = (
            list(clock_port_polarity)[0][1] if clock_port_polarity else None
        )
        input_port_names = [p[0] for p in input_port_width]
        output_port_names = [p[0] for p in output_port_width]

        input_defs = "\n    ".join(
            [f"reg [{width-1}:0] {port}_in ;" for port, width in input_port_width]
        )
        gold_output_defs = "\n    ".join(
            [
                f"wire [{width-1}:0] {port}{self.golden_suffix} ;"
                for port, width in output_port_width
            ]
        )
        gate_output_defs = "\n    ".join(
            [
                f"wire [{width-1}:0] {port}{self.gate_suffix} ;"
                for port, width in output_port_width
            ]
        )
        trigger_assign = (
            "\n    always @(*) begin\n        #5; trigger = ~( "
            + " & ".join(
                [
                    f"{port}{self.golden_suffix} === {port}{self.gate_suffix}"
                    for port in output_port_names
                ]
                + ["1'b1"]
            )
            + " );\n    end\n"
        )

        gold_port_mappings = ",\n        ".join(
            [f".{port}( {port}_in )" for port in input_port_names]
            + [f".{port}( {port}{self.golden_suffix} )" for port in output_port_names]
        )
        gate_port_mappings = ",\n        ".join(
            [f".{port}( {port}_in )" for port in input_port_names]
            + [f".{port}( {port}{self.gate_suffix} )" for port in output_port_names]
        )

        randomize_inputs_lines = "\n            ".join(
            [
                f"{port}_in = {{{', '.join(['$random(seed)']*math.ceil(width/32))}}};"
                for port, width in input_port_width
                if port not in [clock_port_name] + list(reset_port_names)
            ]
        )
        randomize_inputs_task = f"""// task to generate random inputs
    task randomize_inputs;
        begin
            {randomize_inputs_lines}
        end
    endtask
"""
        grouped = {}
        for port, polarity, sync in reset_port_polarity_sync:
            if port not in grouped:
                grouped[port] = []
            grouped[port].append((port, polarity, sync))

        all_reset_combinations = []
        for r in range(1, len(grouped) + 1):
            for ports in combinations(grouped.keys(), r):
                for polarities_syncs in product(*[grouped[port] for port in ports]):
                    all_reset_combinations.append(list(polarities_syncs))
        reset_task_list = []
        for i, reset_comb in enumerate(all_reset_combinations):
            sync_reset_lines = []
            async_reset_lines = []
            unset_lines = []
            for port, polarity, sync in reset_comb:
                if sync:
                    sync_reset_lines.append(f"{port}_in = {polarity};")
                else:
                    async_reset_lines.append(f"{port}_in = {polarity};")
                unset_lines.append(f"{port}_in = {0 if polarity == 1 else 1};")
            reset_lines = (
                (
                    "\n            ".join(sync_reset_lines)
                    + "\n            # 10; toggle_clock; # 10; toggle_clock;\n            "
                    + "\n            ".join(unset_lines)
                )
                if sync_reset_lines
                else "" + "\n            ".join(async_reset_lines + unset_lines)
            )
            reset_task = f"""task reset_{i};
        begin
            {reset_lines}
        end
    endtask
"""
            reset_task_list.append(reset_task)
        directed_tests_task = f"""// Task for directed test. The inputs should be able to activate all functionalities in the golden design, and checks whether the gate design and the golden design are equivalent.
    task directed_tests;
        begin
            // [TODO] directed tests here.
            {'# 10; toggle_clock; # 10; toggle_clock;' if clock_port_name else ''}
        end
    endtask
"""

        toggle_clock_task = f"""// Task to toggle {clock_port_name}_in
    task toggle_clock;
        begin
            {clock_port_name}_in = ~{clock_port_name}_in ;
        end
    endtask
"""
        count_errors_task = f"""// Task to count errors
    task count_errors;
        begin
            if (trigger === 1'b1) begin
                num_errors = num_errors + 1;
            end
            num_all = num_all + 1;
        end
    endtask
"""
        random_reset_lines = "\n            ".join(
            [f"{port}_in = $random(seed);" for port in reset_port_names]
        )

        random_reset_task = f"""// Task for random reset
    task random_reset;
        begin
            {random_reset_lines}
        end
    endtask
"""

        initial_block_lines = [
            "// initial block for random tests and targed tests",
            "initial begin",
            '    if (!$value$plusargs("seed=%d", seed)) seed = 0;',
            f'    if (!$value$plusargs("outerLoopNum=%d", outerLoopNum)) outerLoopNum = {self.random_seq_num};',
            f'    if (!$value$plusargs("innerLoopNum=%d", innerLoopNum)) innerLoopNum = {self.random_seq_steps};',
            (
                f"    {clock_port_name}_in = {0 if clock_port_edge else 1};"
                if clock_port_name
                else ""
            ),
            f"    repeat (outerLoopNum) begin",
            "        random_reset;" if reset_port_names else "",
            "        #100; count_errors;",
            f"        repeat (innerLoopNum) begin",
            "            #100; randomize_inputs;",
            "            #100; toggle_clock;" if clock_port_name else "",
            "            #100; count_errors;",
            "        end",
            "    end",
        ]
        if reset_port_names:
            initial_block_lines.append("    #100;")
            for i in range(len(reset_task_list)):
                initial_block_lines.append(
                    f"    repeat (outerLoopNum) begin",
                )
                initial_block_lines.append(f"        reset_{i};")
                initial_block_lines.append(f"        #100; count_errors;")
                initial_block_lines.append(
                    f"        repeat (innerLoopNum) begin",
                )
                initial_block_lines.append(f"            #100; randomize_inputs;")
                (
                    initial_block_lines.append(f"            #100; toggle_clock;")
                    if clock_port_name
                    else ""
                )
                initial_block_lines.append(f"            #100; count_errors;")
                initial_block_lines.append(f"        end")
                initial_block_lines.append(f"    end")

        initial_block_lines += [
            '    $display("Number of all tests:  %d", num_all);',
            '    $display("Number of errors:     %d", num_errors);',
            '    $display("Error rate: %.8f", num_errors/num_all);',
            "    if (num_errors == 0) begin",
            '        $display("All tests passed.");',
            "    end",
            "    $finish;",
            "end",
        ]
        initial_block = "\n    ".join(initial_block_lines)

        monitor_block = f"""always @(trigger) begin
        if (trigger === 1'b1) begin
            $error("trigger signal is 1, which is not allowed!");
            $finish;
        end
    end
"""

        testbench_code = f"""

module testbench;
    {input_defs}
    {gold_output_defs}
    {gate_output_defs}

    reg trigger;
    real num_all = 0;
    real num_errors = 0;
    integer seed;
    integer outerLoopNum;
    integer innerLoopNum;

    {golden_top}{self.golden_suffix} gold (
        {gold_port_mappings}
    );
    {gate_top}{self.gate_suffix} gate (
        {gate_port_mappings}
    );
    {trigger_assign}
    {toggle_clock_task if clock_port_name else ""}
    {''.join(reset_task_list) if reset_port_names else ""}
    {random_reset_task if reset_port_names else ""}
    {randomize_inputs_task}
    {""}
    {count_errors_task}
    {initial_block}
endmodule
"""
        return testbench_code
    
    def write_code_testbench(
        self,
        golden_path,
        gate_path,
        golden_top,
        gate_top,
        tb_dir,
        input_port_width,
        output_port_width,
        clock_port_polarity,
        reset_port_polarity_sync,
    ):
        if not os.path.exists(tb_dir):
            os.makedirs(tb_dir)
        with open(golden_path, "r") as f:
            golden_code = f.read()
        print("Processing golden code...") if not self.quiet else None
        renamed_golden_code = self.process_verilog(golden_code, self.golden_suffix)

        if gate_path is not None:
            with open(gate_path, "r") as f:
                gate_code = f.read()
            print("Processing gate code...") if not self.quiet else None
            renamed_gate_code = self.process_verilog(gate_code, self.gate_suffix)
            with open(os.path.join(tb_dir, "gate.v"), "w") as f:
                f.write(renamed_gate_code)

        tb_module_code = self.generate_testbench(
            input_port_width,
            output_port_width,
            clock_port_polarity,
            reset_port_polarity_sync,
            golden_top,
            gate_top,
        )
        fim_code = renamed_golden_code + tb_module_code
        with open(os.path.join(tb_dir, "tb.v"), "w") as f:
            f.write(renamed_golden_code)
            f.write("\n")
            f.write(tb_module_code)
        return renamed_golden_code, renamed_gate_code, tb_module_code

    def equiv_with_testbench(
        self,
        golden_path,
        gate_path,
        golden_top,
        gate_top,
        tb_dir,
        port_info=None,
        seed=0,
        outerLoopNum=None,
        innerLoopNum=None,
        timeout=60,
    ):
        if outerLoopNum == None:
            outerLoopNum = self.random_seq_num
        if innerLoopNum == None:
            innerLoopNum = self.random_seq_steps
        assert port_info is not None
        input_port_width, output_port_width, clock_port_polarity, reset_port_polarity_sync = port_info
        
        self.write_code_testbench(
            golden_path=golden_path,
            gate_path=gate_path,
            golden_top=golden_top,
            gate_top=gate_top,
            tb_dir=tb_dir,
            input_port_width=input_port_width,
            output_port_width=output_port_width,
            clock_port_polarity=clock_port_polarity,
            reset_port_polarity_sync=reset_port_polarity_sync,
        )

        command = f"iverilog -g2012 -o {os.path.join(tb_dir,'tb.vvp')} -s testbench {os.path.join(tb_dir,'*.v')} && {os.path.join(tb_dir,f'tb.vvp +seed={seed} +outerLoopNum={outerLoopNum} +innerLoopNum={innerLoopNum}')}"
        res = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        error_rate_pattern = r"Error rate:\s*(\d+\.\d+)\n"
        print(res.stdout.decode("utf-8")) if not self.quiet else None
        if re.search(error_rate_pattern, res.stdout.decode("utf-8")):
            error_rate = float(
                re.search(error_rate_pattern, res.stdout.decode("utf-8")).group(1)
            )
        else:
            error_rate = 1.0
        
        if "All tests passed." in res.stdout.decode("utf-8"):
            print("Test passed!") if not self.quiet else None
            return (
                True,
                error_rate,
                input_port_width,
                output_port_width,
                clock_port_polarity,
                reset_port_polarity_sync,
            )

        else:
            print("Test failed!") if not self.quiet else None
            return (
                False,
                error_rate,
                input_port_width,
                output_port_width,
                clock_port_polarity,
                reset_port_polarity_sync,
            )


class myLogger:
    def __init__(self):
        self.log = []

    def info(self, info_content):
        self.log.append([str(datetime.datetime.now()), "INFO", info_content])

    def debug(self, debug_content):
        self.log.append([str(datetime.datetime.now()), "DEBUG", debug_content])

    def output(self, level):
        match level:
            case "info":
                lines = [l for l in self.log if l[1] == "INFO"]
                text = "\n".join([" - ".join(l) for l in lines])
                return text
            case "debug":
                text = "\n".join([" - ".join(l) for l in self.log])
                return text
            case _:
                raise Exception("Unsupported. Only support info and debug.")